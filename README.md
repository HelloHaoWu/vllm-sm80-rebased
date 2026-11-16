# vLLM SM80 Rebased

An independent integration of **vLLM Backport** with a pinned, newer **vLLM main**
snapshot, validated with GLM-5.3-Flash AWQ W4A16 on four A800 GPUs.

[中文说明](README.zh-CN.md) · [Architecture](docs/backport_rebase/architecture.md) ·
[Build and usage](docs/backport_rebase/build.md) ·
[Validation and limits](docs/backport_rebase/validation.md) ·
[Acknowledgements](ACKNOWLEDGEMENTS.md)

## Credit first

Thank you to the [vLLM team and contributors](https://github.com/vllm-project/vllm)
and to [wtdcode / lazymio and the vLLM Backport contributors](https://github.com/wtdcode/vllm-backport).
Backport supplied the older-GPU adaptations, software FP8, Triton sparse-indexer
scoring, deterministic FP32 Top-K, and many model/performance extensions. vLLM
supplied the serving engine, scheduling/cache infrastructure, model interfaces,
and upstream operator ecosystem.

This repository contributes full-fork migration, compatibility isolation,
integration fixes, and build/model validation. It does not claim the original
Backport kernels as new work. See [full attribution and pinned inputs](ACKNOWLEDGEMENTS.md).
This is not an official release of either upstream project.

## What changed

| Area | Rebased integration |
| --- | --- |
| Engine base | Pinned vLLM main cfd86e43206515956029d5e5cd25ac7bb212da2c |
| Backport input | 29e66dad461b11caae6039f510eaa7913f9dfb19 |
| Operator compatibility | Main _C / _moe_C plus isolated _backport_C |
| Sparse-indexer Top-K | Hardware/dtype-aware selection; preserved SM80 FP32 path |
| Sparse-indexer integration | Retained software FP8 and Triton scoring, adapted metadata/cache interfaces |
| Integration repairs | CUDA-only compiler flags and actual-width KPool remapping |
| Legacy compatibility | Optional W4A8, other models, PP/MTP, LMCache and performance extensions migrated |

The tested GLM model still uses the same W4A16 checkpoint and BF16 attention KV.
A800 does not gain native FP8 Tensor Core computation through software FP8 cache
encoding. Source migration coverage is broader than service-validation coverage.

## Validation snapshot

Matched four-A800 comparisons with 512-token output and 1K/8K/32K input tiers
showed **+1.43% to +4.58% aggregate output throughput with MTP off**, and
**+1.66% to +5.49% with single-token MTP enabled on both stacks**. The paired
campaign had 336 requests and no failed requests. These are net deployment-stack
measurements, including dependency changes; both stacks already support MTP.
A large gain from toggling MTP is not a unique rebase benefit.

Complete SM80 artifacts and real-model API checks were validated privately.
The public branch preserves runtime/build source but removes private notes and
adds publication documentation. It has not been rebuilt into a public wheel.
Formal model-quality evaluation and full validation of every migrated feature
remain incomplete. Read [the scope and measurements](docs/backport_rebase/validation.md)
before interpreting compatibility or performance claims.

## Source release

Build in an isolated Linux/CUDA environment using [this guide](docs/backport_rebase/build.md).
No model weights, credentials, private deployment configuration, binary wheels,
or container images are distributed here. Original Backport Docker Hub tags
are the author's releases, not builds of this integration.

The [Backport README](README.backport.md) and [vLLM README](README.vllm.md) are
retained as upstream reference material. Their support matrices and release
instructions do not override this integration's documented validation scope.

## License

[Apache-2.0](LICENSE), retaining existing copyright and third-party notices.
See [acknowledgements and provenance](ACKNOWLEDGEMENTS.md).
