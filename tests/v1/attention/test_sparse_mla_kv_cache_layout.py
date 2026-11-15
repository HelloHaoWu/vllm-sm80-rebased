# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA allocation and warmup must agree on physical addressing."""

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CacheConfig
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.platforms import current_platform
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.attention.backends.mla.sparse_utils import (
    _CONVERT_REQ_INDEX_TO_GLOBAL_INDEX_KERNEL,
    flat_kv_row_view,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import KVCacheGroup
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import allocate_kv_cache, customize_layer_config

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA sparse MLA")
@pytest.mark.parametrize("buffer_width", [2048, 2176])
@pytest.mark.parametrize("use_graph", [False, True])
def test_triton_sparse_mla_preserves_kpool_tail(buffer_width, use_graph):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.v1.attention.backends.mla.triton_mla_sparse import TritonMLASparseImpl
    from vllm.v1.attention.backends.mla.xpu_mla_sparse import XPUMLASparseMetadata

    device = "cuda"
    indices = torch.full((1, buffer_width), -1, dtype=torch.int32, device=device)
    indices[0, 0] = 0
    indices[0, 2048 if buffer_width > 2048 else 1] = 2048
    cache = torch.zeros((17, 128, 512), dtype=torch.bfloat16, device=device)
    cache[1, 0].fill_(2)
    blocks = torch.arange(17, dtype=torch.int32, device=device).unsqueeze(0)
    blocks[0, 0] = 2
    blocks[0, 16] = 1
    q = torch.zeros((1, 4, 512), dtype=torch.bfloat16, device=device)
    metadata = XPUMLASparseMetadata(
        num_reqs=1,
        max_query_len=1,
        max_seq_len=2049,
        num_actual_tokens=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        slot_mapping=torch.zeros(1, dtype=torch.int64, device=device),
        block_table=blocks,
        req_id_per_token=torch.zeros(1, dtype=torch.int32, device=device),
        block_size=128,
        topk_tokens=2048,
        num_decodes=1,
        num_decode_tokens=1,
    )
    with set_current_vllm_config(VllmConfig()):
        impl = TritonMLASparseImpl(
            num_heads=4,
            head_size=512,
            scale=512**-0.5,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="bfloat16",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            topk_indices_buffer=indices,
            kv_lora_rank=512,
        )
        result, _ = impl.forward_mqa(q, cache, metadata, None)
        torch.testing.assert_close(result, torch.ones_like(q), rtol=0, atol=0)
        if use_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result, _ = impl.forward_mqa(q, cache, metadata, None)
            cache[1, 0].fill_(4)
            graph.replay()
            torch.testing.assert_close(result, torch.full_like(q, 2), rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not current_platform.is_cuda_alike(), reason="CUDA/ROCm only"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "layout", [KVCacheLayout.BLHNC, KVCacheLayout.BLNHC, KVCacheLayout.LBNHC]
)
@pytest.mark.parametrize(
    "backend,cache_dtype,sm100,stride_alignment",
    [
        (AttentionBackendEnum.FLASHINFER_MLA_SPARSE, "auto", True, 32 * 1152),
        (AttentionBackendEnum.FLASHINFER_MLA_SPARSE, "fp8", True, 32 * 576),
        pytest.param(
            AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE_FA4,
            "auto",
            True,
            64 * 1152,
            marks=pytest.mark.skipif(
                not current_platform.is_cuda(), reason="CUDA only"
            ),
        ),
        (AttentionBackendEnum.FLASHMLA_SPARSE, "auto", True, 1152),
        (AttentionBackendEnum.FLASHMLA_SPARSE, "fp8_ds_mla", False, MultipleOf(64)),
        (AttentionBackendEnum.FLASHMLA_SPARSE, "fp8_ds_mla", True, 656),
        (AttentionBackendEnum.FLASHMLA_SPARSE, "nvfp4_ds_mla", True, 352),
    ],
)
def test_allocation_and_warmup_follow_addressing_mode(
    monkeypatch, layout, backend, cache_dtype, sm100, stride_alignment, device
):
    backend = backend.get_class()
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: sm100
    )
    config = SimpleNamespace(
        cache_config=CacheConfig(block_size=64),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        attention_config=SimpleNamespace(hisparse_config=None),
        kernel_config=SimpleNamespace(enable_jit_warmup=True),
    )
    config.cache_config.kv_cache_layout = layout.name
    layer = SimpleNamespace(
        _vllm_config=config,
        attn_backend=backend,
        impl=object.__new__(backend.get_impl_cls()),
        kv_cache_dtype=cache_dtype,
        head_size=576,
        topk_buffer_width=None,
        sliding_window=None,
        indexer=None,
    )
    layer._uses_flat_kv_cache = MethodType(MLAAttention._uses_flat_kv_cache, layer)
    layer_config = customize_layer_config(
        backend, MLAAttention.get_kv_cache_config(layer, config)
    )
    spec = layer_config.spec
    indexer = SimpleNamespace(
        cache_config=config.cache_config, head_dim=132, dtype=torch.uint8
    )
    index_config = DeepseekV32IndexerCache.get_kv_cache_config(indexer, config)
    index_spec = index_config.spec
    assert spec.block_stride_alignment == stride_alignment
    assert index_spec.block_stride_alignment is None
    assert layout in get_supported_kv_cache_layouts(
        # indexer backend has no impl class
        [backend, DeepseekV32IndexerBackend]  # type: ignore[type-abstract]
    )
    specs = {"mla": spec, "indexer": index_spec}
    group = KVCacheGroup.from_layers({"mla": layer_config, "indexer": index_config})
    cache = get_kv_cache_config_from_groups(config, [group], 2**20)
    views = allocate_kv_cache(cache, torch.device(device), layout)
    register = MagicMock()
    monkeypatch.setattr(
        _CONVERT_REQ_INDEX_TO_GLOBAL_INDEX_KERNEL, "register_warmup", register
    )
    MLAAttention.bind_kv_cache(layer, views["mla"])
    DeepseekV32IndexerCache.bind_kv_cache(indexer, views["indexer"])
    assert indexer.kv_cache.stride(0) == views["indexer"].stride(0)
    alignment = spec.get_block_stride_alignment()
    assert views["mla"].stride(0) * views["mla"].element_size() % alignment == 0
    if not layer._uses_flat_kv_cache():
        register.assert_not_called()
    else:
        rows, stride = flat_kv_row_view(layer.kv_cache, 64)
        register.assert_called_once_with(config, block_stride_rows=stride)
        layer.kv_cache[1, 0].fill_(3)
        torch.testing.assert_close(rows[stride], layer.kv_cache[1, 0])
    if alignment == 1:
        assert cache.kv_cache_tensors[0].size == cache.num_blocks * sum(
            item.page_size_bytes for item in specs.values()
        )
