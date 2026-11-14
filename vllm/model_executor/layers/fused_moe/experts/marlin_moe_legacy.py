# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused MoE utilities for GPTQ."""

import math
from collections.abc import Callable

import torch

from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    batched_moe_align_block_size,
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils import marlin_ops_legacy as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils_legacy import (
    marlin_make_workspace_new,
    marlin_moe_intermediate_size,
    marlin_quant_input,
)
from vllm.scalar_type import ScalarType, scalar_types


def _fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: torch.Tensor | None,
    bias2: torch.Tensor | None,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    num_topk: int,
    quant_type: ScalarType,
    apply_router_weight_on_input: bool,
    expert_map: torch.Tensor | None,
    block_size_m: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    activation: MoEActivation = MoEActivation.SILU,
    activation_func: Callable[..., None] | None = None,
    topk_ids: torch.Tensor | None = None,
    input_global_scale1: torch.Tensor | None = None,
    input_global_scale2: torch.Tensor | None = None,
    global_scale1: torch.Tensor | None = None,
    global_scale2: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    intermediate_cache13: torch.Tensor | None = None,
    intermediate_cache2: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    input_dtype: torch.dtype | None = None,
    activation_config: ApplyMoEActivationConfig | None = None,
    mul_topk_weights_in_gemm2: bool = True,
) -> torch.Tensor:
    assert hidden_states.ndim == 2
    M, K = hidden_states.size()
    N = marlin_moe_intermediate_size(w1, w2)
    w13_num_shards = 2 if activation.is_gated else 1
    if workspace is None:
        workspace = marlin_make_workspace_new(hidden_states.device, 4)

    if intermediate_cache13 is None:
        intermediate_cache13 = torch.empty(
            (M * num_topk * max(w13_num_shards * N, K),),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    if intermediate_cache2 is None:
        intermediate_cache2 = torch.empty(
            (M * num_topk, N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

    intermediate_cache1 = _resize_cache(
        intermediate_cache13, (M * num_topk, w13_num_shards * N)
    )

    intermediate_cache3 = _resize_cache(intermediate_cache13, (M * num_topk, K))

    intermediate_cache2 = _resize_cache(intermediate_cache2, (M * num_topk, N))

    a_scales1 = None
    gate_up_input = hidden_states
    if input_dtype == torch.int8:
        gate_up_input, a_scales1 = marlin_quant_input(hidden_states, input_dtype)
        if input_global_scale1 is not None:
            a_scales1 = a_scales1 * input_global_scale1
    elif input_dtype == torch.float8_e4m3fn:
        gate_up_input, a_scales1 = marlin_quant_input(hidden_states, input_dtype)

    intermediate_cache1 = ops.moe_wna16_marlin_gemm(
        gate_up_input,
        intermediate_cache1,
        w1,
        bias1,
        w1_scale,
        a_scales1,
        global_scale1,
        w1_zeros,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=num_topk,
        mul_topk_weights=apply_router_weight_on_input,
        b_q_type=quant_type,
        size_m=M,
        size_n=w13_num_shards * N,
        size_k=K,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )
    activation_input = intermediate_cache1.view(-1, w13_num_shards * N)
    if activation_func is None:
        config = (
            ApplyMoEActivationConfig()
            if activation_config is None
            else activation_config
        )
        apply_moe_activation(
            activation,
            intermediate_cache2,
            activation_input,
            activation_config=config,
            topk_ids=topk_ids,
            expert_map=expert_map,
        )
    else:
        activation_func(
            activation,
            intermediate_cache2,
            activation_input,
            topk_ids=topk_ids,
            expert_map=expert_map,
        )

    if output is None:
        output = intermediate_cache3

    a_scales2 = None
    if input_dtype == torch.int8:
        intermediate_cache2, a_scales2 = marlin_quant_input(
            intermediate_cache2, input_dtype
        )
        if input_global_scale2 is not None:
            a_scales2 = a_scales2 * input_global_scale2
    elif input_dtype == torch.float8_e4m3fn:
        intermediate_cache2, a_scales2 = marlin_quant_input(
            intermediate_cache2, input_dtype
        )

    output = ops.moe_wna16_marlin_gemm(
        intermediate_cache2,
        output,
        w2,
        bias2,
        w2_scale,
        a_scales2,
        global_scale2,
        w2_zeros,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=mul_topk_weights_in_gemm2 and not apply_router_weight_on_input,
        b_q_type=quant_type,
        size_m=M * num_topk,
        size_n=K,
        size_k=N,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    return output


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: torch.Tensor | None,
    bias2: torch.Tensor | None,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: MoEActivation = MoEActivation.SILU,
    activation_func: Callable[..., None] | None = None,
    moe_sum: Callable[..., torch.Tensor | None] | None = None,
    expert_map: torch.Tensor | None = None,
    input_global_scale1: torch.Tensor | None = None,
    input_global_scale2: torch.Tensor | None = None,
    global_scale1: torch.Tensor | None = None,
    global_scale2: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    intermediate_cache13: torch.Tensor | None = None,
    intermediate_cache2: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    input_dtype: torch.dtype | None = None,
    activation_config: ApplyMoEActivationConfig | None = None,
    mul_topk_weights_in_gemm2: bool = True,
) -> torch.Tensor:
    """Run the isolated activation-quantized Marlin MoE path."""
    quant_type = ScalarType.from_id(quant_type_id)
    assert quant_type in [
        scalar_types.uint4,
        scalar_types.uint8b128,
        scalar_types.uint4b8,
        scalar_types.float8_e4m3fn,
        scalar_types.float4_e2m1f,
    ]

    bit4_scalar_types = [
        scalar_types.uint4,
        scalar_types.uint4b8,
        scalar_types.float4_e2m1f,
    ]
    num_bits = 4 if quant_type in bit4_scalar_types else 8

    M, K = hidden_states.size()
    E = w1.size(0)
    topk = topk_ids.size(1)

    # Check constraints.
    assert w1.size(1) * 16 == K, "Hidden size mismatch w1"
    assert w2.size(2) // (num_bits // 2) == K, "Hidden size mismatch w2"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    assert num_bits in [4, 8]
    assert topk_weights.dtype == torch.float32

    if global_num_experts == -1:
        global_num_experts = E
    else:
        # Set M to estimated valid tokens per rank
        M = math.ceil(M * E / global_num_experts)

    # M block size selection logic
    # TODO: tune this further for specific models
    for block_size_m in [8, 16, 32, 48, 64]:
        if M * topk / E / block_size_m < 0.9:
            break

    if input_dtype is not None and input_dtype.itemsize == 1:
        block_size_m = max(block_size_m, 16)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        block_size_m,
        global_num_experts,
        expert_map,
        ignore_invalid_experts=True,
    )

    assert activation is not None
    moe_output = _fused_marlin_moe(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        bias1=bias1,
        bias2=bias2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_topk=topk,
        quant_type=quant_type,
        apply_router_weight_on_input=apply_router_weight_on_input,
        expert_map=expert_map,
        block_size_m=block_size_m,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        activation=activation,
        activation_func=activation_func,
        activation_config=activation_config,
        input_global_scale1=input_global_scale1,
        input_global_scale2=input_global_scale2,
        global_scale1=global_scale1,
        global_scale2=global_scale2,
        w1_zeros=w1_zeros,
        w2_zeros=w2_zeros,
        workspace=workspace,
        intermediate_cache13=intermediate_cache13,
        intermediate_cache2=intermediate_cache2,
        output=None,
        input_dtype=input_dtype,
        mul_topk_weights_in_gemm2=mul_topk_weights_in_gemm2,
    ).view(-1, topk, K)

    if output is None:
        output = torch.empty_like(hidden_states)

    if moe_sum is None:
        if expert_map is not None:
            ops.moe_sum(moe_output, output, topk_ids, expert_map)
            return output
        return torch.sum(moe_output.view(-1, topk, K), dim=1, out=output)
    else:
        return moe_sum(moe_output, output, topk_ids, expert_map)


def batched_fused_marlin_moe(
    hidden_states: torch.Tensor,
    expert_num_tokens: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: torch.Tensor | None,
    bias2: torch.Tensor | None,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: MoEActivation = MoEActivation.SILU,
    expert_map: torch.Tensor | None = None,
    input_global_scale1: torch.Tensor | None = None,
    input_global_scale2: torch.Tensor | None = None,
    global_scale1: torch.Tensor | None = None,
    global_scale2: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    intermediate_cache13: torch.Tensor | None = None,
    intermediate_cache2: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    input_dtype: torch.dtype | None = None,
    activation_func: Callable[..., None] | None = None,
    activation_config: ApplyMoEActivationConfig | None = None,
) -> torch.Tensor:
    """This function massages the inputs so the batched hidden_states can be
    presented as a 2D contiguous tensor that could be used with
    _fused_marlin_moe.

    Note that both batched_fused_marlin_moe and fused_marlin_moe ultimately
    use `ops.moe_wna16_marlin_gemm` for the gemm operation and
    `ops.moe_mna16_marlin_gemm` supports only 2D contiguous hidden_states.
    Note that the moe_align_block_size function indicates,
        - What rows of the A matrix (hidden_states) to access during the
        matmul, via sorted_ids output.
        - What expert_id to use for each block matmul, via expert_ids output.

    In the batched version, the tokens are already grouped/batched by experts
    they subscribe to. Due to this, we can represent the batched hidden_states
    tensor of shape [B, MAX_TOKENS_PER_BATCH, K] as a 2D tensor of shape,
    [B * MAX_TOKENS_PER_BATCH, K]. We may treat this a 2D contiguous tensor
    with topk=1 as each token (row in the tensor) subscribes to exactly one
    expert_id (which is the batch_id). With the expert_num_tokens tensor, that
    indicates how many tokens are actually valid in each batch, the
    batched_moe_align_block_size function constructs the sorted_ids and
    expert_ids tensors, so only relevant/valid rows of A (hidden_states)
    are accessed and are processed with the correct expert_ids.
    """
    assert hidden_states.ndim == 3, (
        f"hidden states must be batched. e.g. [B, MAX_TOKENS, K]."
        f"But got {hidden_states.size()}"
    )

    quant_type = ScalarType.from_id(quant_type_id)
    assert quant_type in [
        scalar_types.uint4,
        scalar_types.uint8b128,
        scalar_types.uint4b8,
        scalar_types.float8_e4m3fn,
        scalar_types.float4_e2m1f,
    ]

    bit4_scalar_types = [
        scalar_types.uint4,
        scalar_types.uint4b8,
        scalar_types.float4_e2m1f,
    ]
    num_bits = 4 if quant_type in bit4_scalar_types else 8

    B, BATCH_TOKENS_MAX, K = hidden_states.size()
    M = hidden_states.view(-1, K).size(0)
    E = w1.size(0)

    # Check constraints.
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    assert expert_num_tokens.size(0) == E
    assert B == E, (
        "Batch must be as big as number of experts as the tokens"
        "are sorted into the batch/expert they belong to"
    )
    assert w1.size(1) * 16 == K, "Hidden size mismatch w1"
    assert w2.size(2) // (num_bits // 2) == K, "Hidden size mismatch w2"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert num_bits in [4, 8]

    # Technically, the tokens are already separated by their expert ids.
    # Hidden-States can just be squeezed to have just 2 dimensions,
    # [B * MAX_TOKENS, K] and top_k can be interpreted as just 1.
    topk = 1

    # TODO(varun) : Choose a decent block size like in fused_marlin_moe
    # Tune block_size_m based on expert capacity to reduce padding overhead.
    block_size_m = 64
    for b_m in [8, 16, 32, 48, 64]:
        if BATCH_TOKENS_MAX / b_m < 0.9:
            block_size_m = b_m
            break

    if input_dtype is not None and input_dtype.itemsize == 1:
        block_size_m = max(block_size_m, 16)

    sorted_token_ids, expert_ids, num_tokens_post_padded = batched_moe_align_block_size(
        max_tokens_per_batch=BATCH_TOKENS_MAX,
        block_size=block_size_m,
        expert_num_tokens=expert_num_tokens,
    )

    # TODO (varun): This can be avoided by plumbing the marlin kernel to
    # ignore topk_weights when topk_weights_ptr is a nullptr.
    topk_weights = torch.ones(
        (M, topk), device=hidden_states.device, dtype=torch.float32
    )

    assert activation is not None
    output = _fused_marlin_moe(
        hidden_states=hidden_states.view(-1, K),
        w1=w1,
        w2=w2,
        bias1=bias1,
        bias2=bias2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        topk_weights=topk_weights,
        num_topk=topk,
        quant_type=quant_type,
        apply_router_weight_on_input=apply_router_weight_on_input,
        activation=activation,
        activation_func=activation_func,
        activation_config=activation_config,
        expert_map=expert_map,
        block_size_m=block_size_m,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        input_global_scale1=input_global_scale1,
        input_global_scale2=input_global_scale2,
        global_scale1=global_scale1,
        global_scale2=global_scale2,
        w1_zeros=w1_zeros,
        w2_zeros=w2_zeros,
        workspace=workspace,
        intermediate_cache13=intermediate_cache13,
        intermediate_cache2=intermediate_cache2,
        output=output.view(-1, K) if output is not None else output,
        input_dtype=input_dtype,
    )

    output = output.view(B, BATCH_TOKENS_MAX, K)

    return output
