# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.layers.quantization.utils import marlin_ops_legacy as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils_legacy import (
    marlin_act_int8_process_scales,
    marlin_moe_padded_intermediate,
    marlin_moe_permute_scales,
    marlin_permute_bias,
    moe_awq_to_marlin_zero_points,
    moe_packed_to_marlin_zero_points,
)


def _pad_w13_shard_cols(x: torch.Tensor, unit: int, padded_unit: int) -> torch.Tensor:
    """Zero-pad each of the two gate/up shards of a `(E, rows, 2 * unit)`
    tensor along its last dim, from `unit` to `padded_unit` columns."""
    if padded_unit == unit:
        return x
    e, rows, _ = x.shape
    x = x.view(e, rows, 2, unit)
    x = torch.nn.functional.pad(x, (0, padded_unit - unit))
    return x.reshape(e, rows, 2 * padded_unit).contiguous()


def _pad_rows(x: torch.Tensor, padded_rows: int) -> torch.Tensor:
    """Zero-pad a `(E, rows, cols)` tensor to `padded_rows` rows."""
    if padded_rows == x.size(1):
        return x
    return torch.nn.functional.pad(x, (0, 0, 0, padded_rows - x.size(1)))


def _pad_w13_bias(bias: torch.Tensor, n: int, padded_n: int) -> torch.Tensor:
    """Zero-pad each gate/up shard of a `(E, 2 * n)` bias to `padded_n`."""
    if padded_n == n:
        return bias
    e = bias.size(0)
    bias = bias.view(e, 2, n)
    bias = torch.nn.functional.pad(bias, (0, padded_n - n))
    return bias.reshape(e, 2 * padded_n).contiguous()


def _process_weights_marlin(
    layer: torch.nn.Module,
    input_dtype: torch.dtype | None,
    num_bits: int,
    pack_factor: int,
    group_size: int,
    w13_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    w13_qzeros: torch.Tensor | None = None,
    w2_qzeros: torch.Tensor | None = None,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,  # w13_qweight
    torch.Tensor,  # w2_qweight
    torch.Tensor,  # w13_scales
    torch.Tensor,  # w2_scales
    torch.Tensor | None,  # w13_qzeros
    torch.Tensor | None,  # w2_qzeros
    torch.Tensor | None,  # w13_input_global_scale
    torch.Tensor | None,  # w2_input_global_scale
    torch.Tensor | None,  # w13_bias
    torch.Tensor | None,  # w2_bias
]:
    """Standard Marlin weight post-processing shared by MARLIN and
    BATCHED_MARLIN backends.

    Steps
    -----
    1. Optional FP8 preprocessing of packed weights / scales.
    2. Repack weights via `gptq_marlin_moe_repack`.
    3. Permute scales (and optionally extract INT8 global scales).
    4. Permute bias tensors.
    """
    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1

    marlin_w13_qweight: torch.Tensor
    marlin_w2_qweight: torch.Tensor
    marlin_w13_scales: torch.Tensor
    marlin_w2_scales: torch.Tensor
    w13_input_global_scale: torch.Tensor | None = None
    w2_input_global_scale: torch.Tensor | None = None
    w13_bias_out: torch.Tensor | None = None
    w2_bias_out: torch.Tensor | None = None

    # --- FP8 weight / scale adjustment ---
    if input_dtype == torch.float8_e4m3fn:
        # NOTE: for non-zp quantization format only
        marlin_w13_qweight = ops.marlin_int4_fp8_preprocess(w13_qweight, inplace=False)
        marlin_w2_qweight = ops.marlin_int4_fp8_preprocess(w2_qweight, inplace=False)
        marlin_w13_scales = w13_scales.data * 512
        marlin_w2_scales = w2_scales.data * 512
    else:
        marlin_w13_qweight = w13_qweight
        marlin_w2_qweight = w2_qweight
        marlin_w13_scales = w13_scales
        marlin_w2_scales = w2_scales

    # --- Pad the intermediate size to a valid Marlin thread tile ---
    # GPTQ packs along K: w13's N is in the (shard) columns, w2's N in the rows.
    N = layer.intermediate_size_per_partition
    padded_N = marlin_moe_padded_intermediate(N, group_size)
    if padded_N != N:
        marlin_w13_qweight = _pad_w13_shard_cols(marlin_w13_qweight, N, padded_N)
        marlin_w2_qweight = _pad_rows(marlin_w2_qweight, padded_N // pack_factor)
        marlin_w13_scales = _pad_w13_shard_cols(marlin_w13_scales, N, padded_N)
        if group_size > 0:
            marlin_w2_scales = _pad_rows(marlin_w2_scales, padded_N // group_size)
        if w13_qzeros is not None:
            w13_qzeros = _pad_w13_shard_cols(
                w13_qzeros, N // pack_factor, padded_N // pack_factor
            )
        if w2_qzeros is not None and group_size > 0:
            w2_qzeros = _pad_rows(w2_qzeros, padded_N // group_size)
        if w13_bias is not None:
            w13_bias = _pad_w13_bias(w13_bias, N, padded_N)

    # --- Repack weights ---
    marlin_w13_qweight = ops.gptq_marlin_moe_repack(
        marlin_w13_qweight,
        marlin_w13_qweight.shape[1] * pack_factor,
        marlin_w13_qweight.shape[2],
        num_bits,
        is_a_8bit=is_a_8bit,
    )
    marlin_w2_qweight = ops.gptq_marlin_moe_repack(
        marlin_w2_qweight,
        marlin_w2_qweight.shape[1] * pack_factor,
        marlin_w2_qweight.shape[2],
        num_bits,
        is_a_8bit=is_a_8bit,
    )

    # --- Permute scales ---
    marlin_w13_scales = marlin_moe_permute_scales(
        s=marlin_w13_scales,
        size_k=layer.intermediate_size_per_partition,
        size_n=marlin_w13_scales.shape[2],
        group_size=group_size,
        is_a_8bit=is_a_8bit,
    )
    group_size_or_pack_factor = group_size if group_size != -1 else pack_factor
    marlin_w2_scales = marlin_moe_permute_scales(
        s=marlin_w2_scales,
        size_k=marlin_w2_scales.shape[1] * group_size_or_pack_factor,
        size_n=marlin_w2_scales.shape[2],
        group_size=group_size,
        is_a_8bit=is_a_8bit,
    )

    if input_dtype == torch.int8:
        if layer.num_groups_w13 > 1:
            marlin_w13_scales, w13_input_global_scale = marlin_act_int8_process_scales(
                marlin_w13_scales
            )
        if layer.num_groups_w2 > 1:
            marlin_w2_scales, w2_input_global_scale = marlin_act_int8_process_scales(
                marlin_w2_scales
            )

    # --- Permute zero points ---
    if w13_qzeros is not None and w2_qzeros is not None:
        w13_qzeros = moe_packed_to_marlin_zero_points(
            w13_qzeros,
            size_k=w13_qzeros.shape[1],
            size_n=w13_qzeros.shape[2] * pack_factor,
            num_bits=num_bits,
            is_a_8bit=is_a_8bit,
        )
        w2_qzeros = moe_packed_to_marlin_zero_points(
            w2_qzeros,
            size_k=w2_qzeros.shape[1],
            size_n=w2_qzeros.shape[2] * pack_factor,
            num_bits=num_bits,
            is_a_8bit=is_a_8bit,
        )

    # --- Permute bias ---
    if w13_bias is not None:
        w13_bias_out = marlin_permute_bias(w13_bias)
    if w2_bias is not None:
        w2_bias_out = marlin_permute_bias(w2_bias)

    return (
        marlin_w13_qweight,
        marlin_w2_qweight,
        marlin_w13_scales,
        marlin_w2_scales,
        w13_qzeros,
        w2_qzeros,
        w13_input_global_scale,
        w2_input_global_scale,
        w13_bias_out,
        w2_bias_out,
    )


def _process_awq_weights_marlin(
    layer: torch.nn.Module,
    weight_bits: int,
    pack_factor: int,
    group_size: int,
    input_dtype: torch.dtype | None,
    w13_qweight: torch.Tensor,
    w2_qweight: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    w13_qzeros: torch.Tensor,
    w2_qzeros: torch.Tensor,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,  # w13_qweight
    torch.Tensor,  # w2_qweight
    torch.Tensor,  # w13_scales
    torch.Tensor,  # w2_scales
    torch.Tensor | None,  # w13_qzeros
    torch.Tensor | None,  # w2_qzeros
    torch.Tensor | None,  # w13_input_global_scale
    torch.Tensor | None,  # w2_input_global_scale
    torch.Tensor | None,  # w13_bias
    torch.Tensor | None,  # w2_bias
]:
    """AWQ-specific Marlin weight post-processing.

    AWQ checkpoints use a different packing order than GPTQ, so they need
    AWQ-specific weight repacking and zero-point conversion before Marlin runs.
    """
    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1
    w13_input_global_scale: torch.Tensor | None = None
    w2_input_global_scale: torch.Tensor | None = None
    w13_bias_out: torch.Tensor | None = None
    w2_bias_out: torch.Tensor | None = None

    if input_dtype == torch.float8_e4m3fn:
        ops.marlin_int4_fp8_preprocess(
            w13_qweight.view(-1, w13_qweight.size(2)),
            w13_qzeros.view(-1, w13_qzeros.size(2)),
            inplace=True,
        )
        ops.marlin_int4_fp8_preprocess(
            w2_qweight.view(-1, w2_qweight.size(2)),
            w2_qzeros.view(-1, w2_qzeros.size(2)),
            inplace=True,
        )
        w13_scales = w13_scales.data * 512
        w2_scales = w2_scales.data * 512

    # --- Pad the intermediate size to a valid Marlin thread tile ---
    # AWQ packs along N: w13's N is in the (shard) columns, w2's N in the rows.
    N = layer.intermediate_size_per_partition
    padded_N = marlin_moe_padded_intermediate(N, group_size)
    if padded_N != N:
        w13_qweight = _pad_w13_shard_cols(
            w13_qweight, N // pack_factor, padded_N // pack_factor
        )
        w2_qweight = _pad_rows(w2_qweight, padded_N)
        w13_scales = _pad_w13_shard_cols(w13_scales, N, padded_N)
        w13_qzeros = _pad_w13_shard_cols(
            w13_qzeros, N // pack_factor, padded_N // pack_factor
        )
        if group_size > 0:
            w2_scales = _pad_rows(w2_scales, padded_N // group_size)
            w2_qzeros = _pad_rows(w2_qzeros, padded_N // group_size)
        if w13_bias is not None:
            w13_bias = _pad_w13_bias(w13_bias, N, padded_N)

    marlin_w13_qweight = ops.awq_marlin_moe_repack(
        w13_qweight,
        size_k=w13_qweight.shape[1],
        size_n=w13_qweight.shape[2] * pack_factor,
        num_bits=weight_bits,
        is_a_8bit=is_a_8bit,
    )
    marlin_w2_qweight = ops.awq_marlin_moe_repack(
        w2_qweight,
        size_k=w2_qweight.shape[1],
        size_n=w2_qweight.shape[2] * pack_factor,
        num_bits=weight_bits,
        is_a_8bit=is_a_8bit,
    )

    marlin_w13_scales = marlin_moe_permute_scales(
        s=w13_scales,
        size_k=layer.intermediate_size_per_partition,
        size_n=w13_scales.shape[2],
        group_size=group_size,
        is_a_8bit=is_a_8bit,
    )
    if input_dtype == torch.int8 and layer.num_groups_w13 > 1:
        marlin_w13_scales, w13_input_global_scale = marlin_act_int8_process_scales(
            marlin_w13_scales
        )

    marlin_w2_scales = marlin_moe_permute_scales(
        s=w2_scales,
        size_k=layer.intermediate_size_per_partition,
        size_n=w2_scales.shape[2],
        group_size=group_size,
        is_a_8bit=is_a_8bit,
    )
    if input_dtype == torch.int8 and layer.num_groups_w2 > 1:
        marlin_w2_scales, w2_input_global_scale = marlin_act_int8_process_scales(
            marlin_w2_scales
        )

    marlin_w13_qzeros = moe_awq_to_marlin_zero_points(
        w13_qzeros,
        size_k=w13_qzeros.shape[1],
        size_n=w13_qzeros.shape[2] * pack_factor,
        num_bits=weight_bits,
        is_a_8bit=is_a_8bit,
    )
    marlin_w2_qzeros = moe_awq_to_marlin_zero_points(
        w2_qzeros,
        size_k=w2_qzeros.shape[1],
        size_n=w2_qzeros.shape[2] * pack_factor,
        num_bits=weight_bits,
        is_a_8bit=is_a_8bit,
    )

    if w13_bias is not None:
        w13_bias_out = marlin_permute_bias(w13_bias)
    if w2_bias is not None:
        w2_bias_out = marlin_permute_bias(w2_bias)

    return (
        marlin_w13_qweight,
        marlin_w2_qweight,
        marlin_w13_scales,
        marlin_w2_scales,
        marlin_w13_qzeros,
        marlin_w2_qzeros,
        w13_input_global_scale,
        w2_input_global_scale,
        w13_bias_out,
        w2_bias_out,
    )
