# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention metadata + slot-mapping helpers for the draft model runner.

Builds the FlashAttention paged-attention metadata and the per-layer
slot_mapping dict that ``unified_kv_cache_update`` reads via
``forward_context.slot_mapping.get(layer_name)``. The mixin expects these
attributes set by the consumer's ``__init__``:

    device, block_size, _block_table_gpu, _draft_vllm_config.
"""

from __future__ import annotations

import torch


class DraftAttnMetadataMixin:
    """Mixin: paged-attention slot mapping + FlashAttentionMetadata."""

    def _get_block_table_tensor(
        self, seq_ids: torch.Tensor | list[int]
    ) -> torch.Tensor:
        """Build a [B, max_blocks] block table tensor from GPU-resident table."""
        if isinstance(seq_ids, list):
            seq_ids_t = torch.tensor(
                seq_ids, dtype=torch.int64, device=self.device
            )
        else:
            seq_ids_t = seq_ids.to(torch.int64)
        # Return a contiguous copy so callers cannot mutate _block_table_gpu
        return self._block_table_gpu[seq_ids_t].contiguous()

    def _compute_slot_mapping(
        self,
        positions: torch.Tensor,
        seq_ids: torch.Tensor | list[int],
    ) -> torch.Tensor:
        """Compute slot mapping using GPU-resident block table (vectorized).

        For each token, the physical slot is:
            physical_block * block_size + offset_in_block
        where:
            logical_block = position // block_size
            offset_in_block = position % block_size
            physical_block = block_table_gpu[seq_id, logical_block]
        """
        if isinstance(seq_ids, list):
            seq_ids_t = torch.tensor(
                seq_ids, dtype=torch.int64, device=self.device
            )
        else:
            seq_ids_t = seq_ids.to(torch.int64)
        logical_blocks = (positions // self.block_size).to(torch.int64)
        offsets = (positions % self.block_size).to(torch.int64)
        physical_blocks = self._block_table_gpu[
            seq_ids_t, logical_blocks
        ].to(torch.int64)
        return physical_blocks * self.block_size + offsets

    def _build_slot_mapping_dict(
        self, slot_mapping: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Build a slot_mapping dict keyed by attention layer names.

        In V1, `unified_kv_cache_update` looks up slot_mapping by layer
        name: `forward_context.slot_mapping.get(layer_name)`.  If the
        key doesn't match, the lookup returns None and the KV cache is
        NOT updated.  We must populate the dict with every attention
        layer's registered name so each layer gets the slot mapping.
        """
        from vllm.model_executor.layers.attention import Attention

        forward_ctx = (
            self._draft_vllm_config.compilation_config.static_forward_context
        )
        mapping: dict[str, torch.Tensor] = {}
        for layer_name, layer in forward_ctx.items():
            if isinstance(layer, Attention):
                mapping[layer_name] = slot_mapping
        return mapping

    def _build_flash_attn_metadata(
        self,
        num_tokens: int,
        seq_lens_tensor: torch.Tensor,
        max_seq_len: int,
        max_query_len: int,
        query_start_loc: torch.Tensor,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """Build FlashAttentionMetadata for forward passes.

        Constructs the minimal metadata needed by the FlashAttention
        backend to perform paged attention with our KV cache.
        """
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

        return FlashAttentionMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens_tensor,
            block_table=block_table,
            slot_mapping=slot_mapping,
            # Cascade attention disabled for draft model
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
            # FA scheduling metadata (None = use FA heuristics)
            scheduler_metadata=None,
            prefix_scheduler_metadata=None,
        )
