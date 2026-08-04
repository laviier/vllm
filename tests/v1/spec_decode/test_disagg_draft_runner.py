# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.worker.gpu.spec_decode.disagg_draft.kv_cache_manager import (
    DraftKVCacheMixin,
)


@pytest.mark.parametrize("stride_order", [(0, 2, 1, 3), (0, 1, 2, 3)])
def test_draft_kv_cache_matches_flash_attention_layout(
    monkeypatch: pytest.MonkeyPatch,
    stride_order: tuple[int, ...],
):
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator,
        "get_memory_info",
        lambda _device: (10_000, 10_000),
    )
    monkeypatch.setattr(
        FlashAttentionBackend,
        "get_kv_cache_stride_order",
        staticmethod(lambda: stride_order),
    )

    runner = DraftKVCacheMixin()
    runner.device = torch.device("cpu")
    runner.dtype = torch.float16
    runner.block_size = 16
    runner.num_kv_heads = 2
    runner.num_layers = 2
    runner.head_dim = 8

    runner._allocate_kv_cache()

    assert runner.kv_caches is not None
    expected_shape = (runner.num_kv_blocks, 2, 16, 16)
    allocation_shape = tuple(expected_shape[i] for i in stride_order)
    inverse_order = tuple(stride_order.index(i) for i in range(len(stride_order)))
    expected_stride = torch.empty(allocation_shape).permute(inverse_order).stride()

    for kv_cache in runner.kv_caches:
        assert kv_cache.shape == expected_shape
        assert kv_cache.stride() == expected_stride
        key_cache, value_cache = kv_cache.transpose(1, 2).split(8, dim=-1)
        assert key_cache.shape == (runner.num_kv_blocks, 16, 2, 8)
        assert value_cache.shape == key_cache.shape
