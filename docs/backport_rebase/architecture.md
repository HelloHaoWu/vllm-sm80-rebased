# Architecture before and after the rebase

This is an integration of vLLM Backport with a pinned vLLM main snapshot, not a
new model architecture. See [acknowledgements](../../ACKNOWLEDGEMENTS.md) for
ownership, source versions, and licensing.

## Layer comparison

| Layer | Previous deployment | Rebased integration |
| --- | --- | --- |
| Engine | Older upstream engine with Backport extensions | Pinned main engine with adapted Backport extensions |
| Model integration | Older model and metadata interfaces | Current model layout, query-shard metadata, cache and graph-buffer interfaces |
| Native operators | Older registrations and call contracts | Current _C / _moe_C plus isolated _backport_C compatibility registrations |
| Sparse indexer | Backport software FP8, Triton scoring and deterministic FP32 Top-K | Same fallback capabilities adapted to current interfaces, with explicit Top-K dispatch |
| Quantized experts | Existing W4A16 checkpoint and Marlin execution | Same model weights; main W4A16 path plus isolated optional legacy W4A8 |
| Runtime validation | Previously fixed Backport serving baseline | Complete SM80 artifact build, integration fixes and real four-A800 validation |

The serving path remains client API, scheduler/model runner, model execution,
then supported operator backends. GLM's DSA, KDA, mHC, and MTP components are not
a single sequential pipeline implied by the rows above.

## What remains the same

- The tested model is GLM-5.3-Flash AWQ W4A16, on four A800 GPUs with TP4.
- Routed-expert weights use the existing INT4 weight representation and
  Marlin/WNA16 execution path; this rebase does not re-quantize the checkpoint.
- Main attention KV cache is BF16 in the tested deployment.
- Sparse-indexer cache can use software-encoded FP8 bytes; this is distinct
  from the main attention KV cache. A800 does not gain native FP8 Tensor Core
  arithmetic through this software fallback.
- Both old and new stacks support MTP. The normal deployment kept MTP disabled;
  single-token MTP was tested separately on both versions.

## Compatibility boundary and source map

| Responsibility | Source entry |
| --- | --- |
| Native compatibility registration | csrc/backport/ and vllm/_backport_ops.py |
| Hardware/dtype-aware Top-K dispatch | vllm/model_executor/layers/indexer_topk.py |
| SM80 software FP8 | vllm/v1/attention/ops/fp8_sm80.py |
| Triton indexer logits | vllm/v1/attention/ops/mqa_logits_triton.py |
| GLM sparse indexer integration | vllm/models/glm5next/nvidia/sparse_indexer.py |
| KPool tail and remapping regression | tests/v1/attention/test_kpool_tail_slot_mapping.py |
| User-visible compatibility settings | docs/features/quantization/backport_sm80.md |

Top-K auto dispatch preserves Backport's FP32 implementation on SM80. BF16
scores and newer GPUs enter the mainline selection chain, subject to each
backend's constraints. Forced Backport dispatch requires CUDA SM80+, FP32,
contiguous columns, and K of 512, 1024, or 2048. Unsupported forced combinations
raise rather than silently casting scores. Selection is based on hardware,
dtype, and static constraints, so CUDA graph capture/replay follows a stable
path. Newer-GPU policies are present in source but were not validated on hardware
in this integration's A800 campaign.

Legacy dense/MoE Marlin W4A8 is isolated from the main W4A16 path. It is an
optional compatibility capability, not the quantization mode of the tested
GLM service. Inherited FP8 activation modes still require supported newer
hardware; they do not imply native FP8 computation on SM80.

## Integration fixes

The full fork migration required adaptation across model interfaces, mHC,
DSpark, PP/MTP, shared experts, LMCache build options, and KV offload reference
handoff. This records source coverage, not complete service-level coverage for
every inherited feature.

Two issues appeared during full builds and real-model warmup:

1. A CUDA-specific compiler flag was also reaching C++ compilation. It now
   applies only to CUDA translation units.
2. Sparse-MLA remapping received the configured 2048 selection budget even when
   KPool tails and alignment produced a 2176-column buffer. Remapping now receives
   the actual buffer width, retaining tail entries and invalid -1 padding. The
   selection budget and selection algorithm are unchanged.

The latter is separate from the earlier Backport KPool addressing/stride/RING
repair already present in the old baseline. See the regression test above.

The integration follows current upstream's strict Anthropic thinking validation
and CPU-offload defaults for PLE/Engram, while retaining documented explicit
configuration compatibility. These defaults do not mean the tested GLM model
was entirely offloaded to CPU.

An inherited mHC fusion option referenced a function absent from the source
fork. The integration warns and uses the supported post-mix plus separate
reduction; it does not claim that missing fusion as a retained speedup.

## Interpretation

The contribution is preserving and adapting an existing older-GPU capability
set on a newer engine, isolating incompatible legacy operators, and validating
real integration contracts. It is not the invention of Backport's software FP8,
Triton scoring, or deterministic Top-K. Performance and quality claims are
limited to the [recorded validation](validation.md).
