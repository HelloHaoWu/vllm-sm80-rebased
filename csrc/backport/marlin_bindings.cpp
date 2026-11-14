// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/csrc/stable/library.h>
STABLE_TORCH_LIBRARY_FRAGMENT(_backport_C, ops) {
  ops.def(
      "marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, Tensor? "
      "b_bias_or_none,Tensor b_scales, Tensor? a_scales, Tensor? global_scale, "
      "Tensor? b_zeros_or_none, Tensor workspace, int b_type_id, SymInt "
      "size_m, SymInt size_n, SymInt size_k, bool use_atomic_add, bool "
      "use_fp32_reduce, bool is_zp_float) -> Tensor");
  ops.def(
      "gptq_marlin_repack(Tensor b_q_weight, SymInt size_k, SymInt size_n, int "
      "num_bits, bool is_a_8bit) -> Tensor");
  ops.def(
      "awq_marlin_repack(Tensor b_q_weight, SymInt size_k, SymInt size_n, int "
      "num_bits, bool is_a_8bit) -> Tensor");
  ops.def(
      "marlin_int4_fp8_preprocess(Tensor qweight, Tensor? qzeros_or_none, bool "
      "inplace) -> Tensor");
  ops.def(
      "moe_wna16_marlin_gemm(Tensor! a, Tensor? c_or_none,Tensor! b_q_weight, "
      "Tensor? b_bias_or_none,Tensor! b_scales, Tensor? a_scales, Tensor? "
      "global_scale, Tensor? b_zeros_or_none,Tensor! workspace,Tensor "
      "sorted_token_ids,Tensor! expert_ids, Tensor! "
      "num_tokens_past_padded,Tensor! topk_weights, int moe_block_size, int "
      "top_k, bool mul_topk_weights, int b_type_id,int size_m, int size_n, int "
      "size_k, bool use_atomic_add,bool use_fp32_reduce, bool is_zp_float,int "
      "thread_k, int thread_n, int blocks_per_sm) -> Tensor");
}
