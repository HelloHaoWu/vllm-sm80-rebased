// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "ops.h"
#include <torch/csrc/stable/library.h>

void backport_persistent_topk(const torch::stable::Tensor&,
                              const torch::stable::Tensor&,
                              torch::stable::Tensor&, torch::stable::Tensor&,
                              int64_t, int64_t);
void backport_top_k_per_row_decode(const torch::stable::Tensor&, int64_t,
                                   const torch::stable::Tensor&,
                                   torch::stable::Tensor&, int64_t, int64_t,
                                   int64_t, int64_t);
void backport_top_k_per_row_prefill(const torch::stable::Tensor&,
                                    const torch::stable::Tensor&,
                                    const torch::stable::Tensor&,
                                    torch::stable::Tensor&, int64_t, int64_t,
                                    int64_t, int64_t);

STABLE_TORCH_LIBRARY_FRAGMENT(_backport_C, ops) {
  ops.def(
      "persistent_topk(Tensor logits, Tensor lengths, Tensor! output, "
      "Tensor workspace, int k, int max_seq_len) -> ()");
  ops.def(
      "top_k_per_row_prefill(Tensor logits, Tensor starts, Tensor ends, "
      "Tensor! output, int rows, int stride0, int stride1, int k) -> ()");
  ops.def(
      "top_k_per_row_decode(Tensor logits, int next_n, Tensor lengths, "
      "Tensor! output, int rows, int stride0, int stride1, int k) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_backport_C, CUDA, ops) {
  ops.impl("persistent_topk", TORCH_BOX(&backport_persistent_topk));
  ops.impl("top_k_per_row_prefill", TORCH_BOX(&backport_top_k_per_row_prefill));
  ops.impl("top_k_per_row_decode", TORCH_BOX(&backport_top_k_per_row_decode));
}
