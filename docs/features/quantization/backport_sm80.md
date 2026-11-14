# Backport compatibility on current upstream

The migration pins upstream cfd86e43206515956029d5e5cd25ac7bb212da2c and fork
29e66dad461b11caae6039f510eaa7913f9dfb19. It is an integration candidate until
operator and model validation is complete.

## Sparse indexer Top-K

`--kernel-config.sparse-indexer-topk-backend` (or the equivalent JSON field)
accepts `backport` in addition to current upstream values. `auto` uses the
preserved FP32 deterministic kernels on SM80 (A100/A800), and current upstream
selection on newer devices or BF16 inputs. Manual `backport` requires CUDA
SM80+, FP32 scores with contiguous columns, and k in 512/1024/2048. Unsupported
forced combinations raise an error. There is no implicit score dtype cast.

The old operators live in `_backport_C`, without overriding current `_C`
schemas. Selection uses device capability, dtype and static shape, not score
contents or live sequence lengths, so the captured path stays fixed during
CUDA graph replay. Eager calls log device, dtype and resolved backend. The
Backport path preserves deterministic tie selection and output order; upstream
paths retain their own semantics.

## Defaults and compatibility limits

PLE/Engram defaults to CPU offload as in current upstream. Explicit
`engram_config.cpu_offload` overrides the legacy `VLLM_PLE_CPU_OFFLOAD=0|1`
environment alias; without either override the default is True.

Anthropic thinking payloads use current upstream strict validation. Enabled
thinking requires budget_tokens >= 1024; unknown thinking types are rejected.

`VLLM_MHC_POST_FUSE_SQRSUM` existed in the fork but referenced a missing fused
kernel. This candidate warns and uses the supported post-mix plus separate
reduction. It does not claim the missing fused speedup. Other mHC paths remain.

Single-A800 operator testing cannot validate multi-GPU PP/TP/EP/MTP, LMCache
services, other GPU architectures, or full model accuracy. Record those limits
alongside validation results before deployment.
