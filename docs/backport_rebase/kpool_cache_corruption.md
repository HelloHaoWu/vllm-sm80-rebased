# GLM KPool 缓存寻址错误：历史问题、修复来源与当前状态

## English summary

The historical Backport deployment used an NVIDIA KPool prefill tail-seed kernel
that assumed densely packed tail blocks. The runtime actually stored them with
the indexer cache's padded stride. Misplaced writes could silently corrupt sparse
indexer keys/scales and therefore Top-K selection, while leaving the main MLA KV
data untouched. The addressing repair comes from
[vLLM PR #57477](https://github.com/vllm-project/vllm/pull/57477), already inherited
by this rebased branch. The later 2048/2176 index-buffer-width integration repair
is a separate issue. This note describes an old deployment baseline, not every
version of vLLM or Backport.

## 1. 问题发生在哪个环节

我们早期部署所用的旧 Backport 基线包含这项缺陷。具体位置是 GLM-5.3-Flash 在 NVIDIA 路径上，用于初始化 KPool 尾部缓存的 prefill 内核 _kpool_tail_seed_kernel。

理解影响范围，需要区分三部分数据：

| 数据 | 用途 | 与这项缺陷的关系 |
| --- | --- | --- |
| KPool 尾部缓存 | 保存尚未形成完整池的原始索引 K 和 gate/score 数据 | 本应写入的目标 |
| DSA 索引缓存 | 保存池化后的索引 K 和缩放信息，用于历史位置打分及 Top-K | 可能被错误写入覆盖 |
| 主注意力 KV 缓存 | 保存选中位置真正参与 MLA 注意力计算的数据 | 该错误的写入没有直接覆盖它 |

索引缓存决定后续注意力读取哪些历史位置。即使主注意力 KV 数据仍在，索引数据被破坏，也可能让模型选错上下文位置并产生异常回答。

## 2. 根因：将有间隔的缓存块当成密集数组

运行时，tail 缓存是与 indexer 缓存共享底层分配的张量视图，块间距按 indexer 页大小设置。正常情况下，两类缓存分配不同的块编号，不会相互覆盖；访问视图时必须使用实际 stride。

旧 NVIDIA seed 内核直接按逻辑形状计算每块大小，假定尾部缓存块紧密排列，没有读取 tail 张量真实的块 stride。结果是本请求的尾部缓存可能没有正确初始化，数据却写进了其他 indexer 块的 key 字节或 scale 区域。

下面是上游 PR 使用的布局示例，均为字节；具体数值随配置变化，不能作为所有部署的常量：

| 项目 | 数值 |
| --- | --- |
| 每块尾部有效数据 | 2048 字节 |
| 实际缓存块起点间距 | 38016 字节 |
| 块号 18 的正确起点 | 18 × 38016 = 684288 |
| 旧代码算出的起点 | 18 × 2048 = 36864 |

因此，这项缺陷属于缓存写地址计算错误。是否使用 FP8 存储、BF16 主注意力 KV 或 W4A16 权重，不能单独解释或修复这个地址错误。它涉及 NVIDIA 路径，不局限于 A800。

## 3. 为什么可能看起来像“模型忘了任务”

模型服务可能继续返回 HTTP 成功，原始输入也确实包含完整任务；但后续稀疏注意力使用了受损的索引数据，选错了需要读取的位置。异常可能发生在模型语义表现上，而没有对应的接口报错。

上游复现显示，后续请求可以持续破坏早期缓存提示词的索引页。前缀缓存继续命中时，这些页不会因命中而自动重新计算，因此问题可能积累，并表现为查错信息、漏读上下文或误判用户没有提出任务。上游同时提供了缓存内容被修改的直接检查和回答异常的复现；这比仅凭模型回答推断原因更具体。

我们早期排查也观察过“任务已经发送，模型却认为没有任务”的行为，并确认所用源码含有相应寻址缺陷。不过，不能由此把每次异常回答或所有评测低分都归因于它。

为诊断而使用新缓存块、cache_salt 或绕开旧缓存，可能缓解某些样例；这些做法不会修正错误的写入地址。

## 4. 修复来源与本仓库实现

修复来自 vLLM 上游贡献者的 [PR #57477](https://github.com/vllm-project/vllm/pull/57477)，已于 2026-09-20 合并，对应提交为 [db1bfdd4fb](https://github.com/vllm-project/vllm/commit/db1bfdd4fb0dd7b8226402ee00abc7a987561b7c)。我们先将其移植到旧部署，完整 rebase 时保留了主线中的修复。该寻址修复不作为本仓库的原创算法或原创发现。

核心变化是向内核传入 tail 张量的真实步长：

```
TAIL_BLOCK_ELEMS = tail_kv_cache.stride(0)
KPOOL_HEAD = tail_kv_cache.stride(1)
```

块起点按 block_id × TAIL_BLOCK_ELEMS 定位，K 与 score 分量之间的偏移使用 KPOOL_HEAD。当前主线还包含后续 tail ring 等布局演进；阅读当前代码时应沿用其实际布局，不能把旧版本的密集寻址表达式移植回来。

当前实现入口：[kpool_compress.py](../../vllm/models/glm5next/nvidia/ops/kpool_compress.py)。

对应既有回归：[test_prefill_seed_honors_padded_tail_block_stride](../../tests/kernels/test_kpool_decode_update_batched.py)。

在已具备匹配依赖、CUDA 和 GPU 的独立验证环境中，可定向运行：

```
.venv/bin/python -m pytest tests/kernels/test_kpool_decode_update_batched.py -q -k test_prefill_seed_honors_padded_tail_block_stride
```

它检查有填充间距的 tail 视图是否被正确初始化。本次文档补充没有运行这个 GPU 命令；历史部署时的独立专项回归已通过。

## 5. 与 rebase 后的 2048/2176 问题区分

| 对比项 | 历史 stride 寻址缺陷 | rebase 集成时的缓冲宽度不匹配 |
| --- | --- | --- |
| 发生位置 | prefill 尾部缓存写入 | 稀疏 MLA 索引到物理位置的转换调用 |
| 错误假设 | 尾部缓存块紧密排列 | 配置选择预算等于实际索引张量宽度 |
| 直接表现 | 写错地址，可能静默破坏索引缓存 | 实际 2176 列却传入 2048，预热触发形状断言 |
| 修复方式 | 使用真实块和分量 stride | 传入实际缓冲宽度，保留尾部及 -1 对齐槽 |
| 来源 | vLLM 上游 PR 57477，先移植后由 rebase 继承 | 本次完整框架迁移中的集成修复 |

后者的运行代码及真实注意力/CUDA graph 回归已包含在本仓库中，见 [KPool 尾部与索引转换测试](../../tests/v1/attention/test_kpool_tail_slot_mapping.py)。它没有把 Top-K 选择预算从 2048 改成 2176；2176 是包含尾部和对齐槽的实际缓冲宽度。

## 6. 当前状态与结论边界

公开分支同时包含上游 stride 寻址修复和后来的实际宽度修复。旧新版 TPS 对照中的旧部署也已经应用 stride 补丁，因此不能把“有 bug 的旧版”与“修好的新版”混在一起，作为 rebase 的性能或质量提升。

历史专项回归、真实模型检查和当前公开源码身份的说明见 [验证记录](validation.md)。这些结果没有证明所有模型质量问题都已解决；本仓库尚未完成新版正式 UnityBench/ABV 质量回归。本次仅补充问题说明和阅读入口，没有修改运行代码，也没有重新操作或测试在线模型服务。
