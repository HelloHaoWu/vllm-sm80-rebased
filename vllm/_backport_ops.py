# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Meta implementations for the isolated Backport operator namespace."""

import torch


def _register(name, fn):
    if hasattr(torch.ops._backport_C, name):
        torch.library.register_fake(f"_backport_C::{name}")(fn)


def _topk_fake(*args):
    return None


def _repack_fake(weight, size_k, size_n, num_bits, is_a_8bit):
    return weight.new_empty((size_k // 16, size_n * 16 // (32 // num_bits)))


def _preprocess_fake(qweight, qzeros_or_none=None, inplace=False):
    return qweight if inplace else torch.empty_like(qweight)


def _gemm_fake(
    a,
    c,
    weight,
    bias,
    scales,
    a_scales,
    global_scale,
    zeros,
    workspace,
    b_type,
    size_m,
    size_n,
    size_k,
    *flags,
):
    if c is not None:
        return c
    dtype = a.dtype if a.dtype in (torch.float16, torch.bfloat16) else scales.dtype
    return torch.empty((size_m, size_n), device=a.device, dtype=dtype)


def _moe_fake(
    a,
    c,
    weight,
    bias,
    scales,
    a_scales,
    global_scale,
    zeros,
    workspace,
    sorted_ids,
    expert_ids,
    padded_tokens,
    topk_weights,
    block_size,
    top_k,
    mul_weights,
    b_type,
    size_m,
    size_n,
    size_k,
    *flags,
):
    if c is not None:
        return c
    dtype = a.dtype if a.dtype in (torch.float16, torch.bfloat16) else scales.dtype
    return torch.empty((size_m * top_k, size_n), device=a.device, dtype=dtype)


for _name in ("persistent_topk", "top_k_per_row_prefill", "top_k_per_row_decode"):
    _register(_name, _topk_fake)
for _name in ("gptq_marlin_repack", "awq_marlin_repack"):
    _register(_name, _repack_fake)
_register("marlin_int4_fp8_preprocess", _preprocess_fake)
_register("marlin_gemm", _gemm_fake)
_register("moe_wna16_marlin_gemm", _moe_fake)
