# vLLM SM80 Rebased

将 vLLM Backport 的完整增量迁移到固定的较新 vLLM main 快照，并在 A800 上验证 GLM-5.3-Flash AWQ W4A16 的独立集成版本。

首先感谢 [vLLM](https://github.com/vllm-project/vllm) 团队及贡献者，以及 [vLLM Backport](https://github.com/wtdcode/vllm-backport) 作者 wtdcode / lazymio 及其贡献者。原有 SM80 适配、软件 FP8、Triton 索引器和确定性 Top-K 来自 Backport。我们的工作是迁移、兼容层组织、冲突适配、集成修复、构建与验证。完整归属见 [致谢与来源](ACKNOWLEDGEMENTS.md)。

## 这次升级改变了什么

| 范围 | 原 Backport 部署 | 融合新版 |
| --- | --- | --- |
| 框架基础 | 较旧 vLLM 骨架及 fork 扩展 | 固定 vLLM main cfd86e432065 加 Backport 增量 |
| 算子入口 | 旧接口及调用链 | 主线 _C / _moe_C 与隔离的 _backport_C |
| Top-K | 原 Backport 路径 | SM80 FP32 默认保留实现；BF16 和更新硬件进入主线选择链 |
| DSA | 软件 FP8 与 Triton 回退 | 保留回退，适配新版 metadata、页表、图缓冲和索引宽度 |
| Marlin | 原 W4A16 及可选 W4A8 | W4A16 使用主线路径，旧 W4A8 单独隔离 |
| 其他扩展 | 多模型、PP/MTP、LMCache 和性能扩展 | 一起迁移源码；未全部完成服务级验收 |

这次没有改变模型结构、W4A16 checkpoint 或 BF16 主注意力 KV 缓存，也没有让 A800 获得原生 FP8 Tensor Core。FP8 索引缓存和主注意力 KV 缓存用途不同。

完整架构与源码入口见 [架构对比](docs/backport_rebase/architecture.md)。

## 具体集成修复

1. 将旧 Top-K / Marlin W4A8 兼容算子隔离，避免覆盖当前主线注册与 schema。
2. 让 NVCC 专有参数仅作用于 CUDA 源文件，避免把该参数传给 GCC 编译的 C++ 文件。
3. 物理索引转换使用实际缓冲区宽度，保留 KPool 尾部 token 和 -1 对齐槽。2048 是选择预算，实际对齐缓冲区可能为 2176，不能混用。
4. 采用主线 Anthropic thinking 严格校验、PLE/Engram 默认 CPU offload，并保留已经确认的显式配置兼容。

配置细节见 [SM80 与旧 Marlin 兼容说明](docs/features/quantization/backport_sm80.md)。

## 旧部署中的 KPool 缓存问题

早期基线中的 NVIDIA prefill 内核没有按真实块间距写入 KPool 尾部缓存，可能覆盖 DSA 索引缓存，造成 Top-K 选错上下文。寻址修复来自 vLLM 上游 PR #57477，本分支已继承。它与 rebase 后“实际缓冲区 2176 列、调用仍传 2048”的集成问题不同；两项修复都已包含，不能据此宣称全部评测低分已解决。

具体机制、地址示例、修复来源和验证边界见 [KPool 缓存问题说明](docs/backport_rebase/kpool_cache_corruption.md)。

## 验证与性能

在同一组四张 A800、同一份 GLM-5.3-Flash AWQ W4A16 权重和相同配置下，与原 Backport 部署比较：关闭 MTP 时总输出吞吐净提升约 1.43%～4.58%；双方均开启单 token MTP 时约提升 1.66%～5.49%。这是整套部署栈的变化，包含 Torch/Triton 等依赖变化，不能单独归因于某个算子或 Git 操作。

两边本来都支持 MTP。MTP 开关产生的较大收益不能记作 rebase 独有收益。

完整制品和真实四卡模型协议检查已在私有验证环境完成；最终索引宽度修复另有 4 项真实注意力和 CUDA graph 回归通过。基础构建曾有 375 项通过、1 项 ROCm 条件跳过；不是把所有阶段测试都在最终包重新执行了一遍。

公开版保留相同的运行源码，清除了私有部署记录；提交身份和由 Git 派生的版本标签会变化。公开提交本身尚未重新构建 wheel，未发布可下载的二进制或 Docker 镜像。正式 UnityBench/ABV 得分尚未重跑，不能宣称模型得分不下降。具体数据和边界见 [验证记录](docs/backport_rebase/validation.md)。

## 构建与使用

这是源码发布。请先阅读 [构建与使用说明](docs/backport_rebase/build.md)，在独立 Linux/CUDA 环境中构建。不要把原作者的 Docker Hub 镜像当作本仓库融合新版镜像。

保留了 [Backport 原始 README](README.backport.md) 与 [vLLM 原始 README](README.vllm.md)，供溯源与学习；其中的支持矩阵和镜像标签属于相应上游，不能替代本集成版本的验收范围。
