<div align="center">

# vLLM SM80 Rebased

### A newer vLLM base for the SM80 Backport stack

An independent integration of [vLLM Backport](https://github.com/wtdcode/vllm-backport) with a pinned, newer [vLLM](https://github.com/vllm-project/vllm) main snapshot, validated with **GLM-5.3-Flash AWQ W4A16 on 4 × NVIDIA A800**.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Validated GPU: 4 × A800](https://img.shields.io/badge/validated-4%20%C3%97%20A800-76b900)
![Distribution: source](https://img.shields.io/badge/distribution-source%20only-orange)

[中文说明](README.zh-CN.md) · [Architecture](docs/backport_rebase/architecture.md) · [Build & serve](docs/backport_rebase/build.md) · [Validation](docs/backport_rebase/validation.md) · [Acknowledgements](ACKNOWLEDGEMENTS.md)

</div>

---

## At a glance

| | |
| --- | --- |
| **vLLM base** | Pinned `main` snapshot: [`cfd86e432065`](https://github.com/vllm-project/vllm/commit/cfd86e43206515956029d5e5cd25ac7bb212da2c) |
| **Backport input** | [`29e66dad461b`](https://github.com/wtdcode/vllm-backport/commit/29e66dad461b11caae6039f510eaa7913f9dfb19) |
| **Validated configuration** | 4 × A800, GLM-5.3-Flash AWQ W4A16, tensor parallelism 4, BF16 attention KV |
| **Availability** | Source checkout; no wheel or container image is published for this integration |

This fork migrates the Backport additions onto the newer engine, isolates compatibility operators, and repairs integration issues found during the rebase. It is **not an official release** of vLLM or vLLM Backport. Migrated source coverage exceeds the configurations validated as a service.

## What this integration changes

| Area | Rebased integration |
| --- | --- |
| Native operators | Keeps mainline `_C` / `_moe_C` while isolating older compatibility operators in `_backport_C`. |
| Sparse indexer | Retains software FP8 encoding and Triton scoring while adapting metadata, cache, and buffer interfaces. |
| Top-K selection | Preserves the deterministic SM80 FP32 path with hardware and dtype aware selection. |
| Build and indexing | Limits NVCC-only flags to CUDA sources and uses the actual KPool index-buffer width. |
| Other Backport additions | Migrates optional W4A8, model, PP/MTP, LMCache, and performance extensions in source; their service validation varies. |

The tested model keeps its W4A16 checkpoint and BF16 attention KV. Software FP8 cache encoding does not give A800 native FP8 Tensor Core computation. See the [architecture notes](docs/backport_rebase/architecture.md) and [SM80 compatibility guide](docs/features/quantization/backport_sm80.md) for the implementation details.

## Measured on four A800 GPUs

In matched comparisons against the older Backport deployment, aggregate output throughput improved **1.43%–4.58% with MTP off** and **1.66%–5.49% with single-token MTP on in both stacks**. The campaign covered 1K, 8K, and 32K input tiers, 512 output tokens per request, and 336 requests with no failures.

These are end-to-end deployment-stack measurements, including dependency changes. They do not isolate the effect of the rebase or any one kernel. Both stacks already supported MTP. The [validation report](docs/backport_rebase/validation.md) contains the per-tier results, test environment, regressions, and limits.

## Build and serve

This repository distributes **source only**. Follow the [build and usage guide](docs/backport_rebase/build.md) in an isolated Linux/CUDA environment before running the integration. The guide records the required toolchain, core wheel-build command, and a four-GPU serving example.

The Docker Hub tags in the original Backport documentation belong to its authors; they are not builds of this rebased integration. No model weights, credentials, private deployment configuration, wheel, or container image are distributed here.

## Validation boundaries

Complete SM80 artifacts and real-model API checks were validated privately. The public branch keeps the runtime and build source, but its public commit has **not** been rebuilt into a downloadable wheel. Formal model-quality evaluation and full service validation of every migrated feature remain incomplete. The configured context limit is not evidence of a one-million-token input test.

An earlier deployment baseline also had a [KPool prefill cache-addressing defect](docs/backport_rebase/kpool_cache_corruption.md), fixed upstream in [vLLM PR #57477](https://github.com/vllm-project/vllm/pull/57477) and inherited here. That fix is distinct from this rebase's index-buffer-width repair.

## Credits and license

Thanks to the [vLLM team and contributors](https://github.com/vllm-project/vllm) and to [wtdcode / lazymio and the vLLM Backport contributors](https://github.com/wtdcode/vllm-backport). The older-GPU adaptations, software FP8, Triton sparse-indexer scoring, deterministic FP32 Top-K, and many model and performance extensions originate in Backport. This repository contributes the full-fork migration, compatibility isolation, integration fixes, and build/model validation; it does not claim the original Backport kernels as new work.

The [original Backport README](README.backport.md) and [vLLM README](README.vllm.md) are retained as references. Their support matrices and release instructions do not define this integration's validation scope. See [acknowledgements and pinned inputs](ACKNOWLEDGEMENTS.md) for full provenance. Licensed under [Apache-2.0](LICENSE), retaining existing copyright and third-party notices.
