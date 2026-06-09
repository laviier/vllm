# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache and block-allocator mixin for the draft model runner.

Separates block allocation, dedicated-block reservation per verify
server, and branch-table swapping from the forward-pass code in
DraftModelRunner. The mixin expects these attributes set by the
consumer's ``__init__``:

    device, dtype, block_size, num_kv_heads, num_layers, head_dim,
    max_model_len, max_num_blocks, max_num_seqs, vllm_config,
    _draft_vllm_config (set during load_model), _block_table_gpu,
    _block_tables, _dedicated_blocks_by_vs, _seq_lens, _free_list,
    _next_free_block, num_kv_blocks, kv_caches.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class DraftKVCacheMixin:
    """Block-allocator and KV-cache management for DraftModelRunner."""

    def _allocate_kv_cache(self) -> None:
        """Allocate KV cache on the draft GPU."""
        torch.cuda.empty_cache()

        free_mem, _ = torch.cuda.mem_get_info(self.device)
        usable_bytes = int(free_mem * 0.8)

        bytes_per_block = (
            2 * self.num_layers * self.block_size
            * self.num_kv_heads * self.head_dim
            * torch.finfo(self.dtype).bits // 8
        )

        self.num_kv_blocks = max(1, usable_bytes // bytes_per_block)

        logger.info(
            "Draft KV cache: %d blocks × %d tokens/block = %d max tokens, "
            "%.1f GB",
            self.num_kv_blocks, self.block_size,
            self.num_kv_blocks * self.block_size,
            self.num_kv_blocks * bytes_per_block / 1e9,
        )

        # Layout: (num_blocks, 2, block_size, num_kv_heads, head_dim).
        # Matches main's FlashAttention backend after #42095, which
        # unpacks key/value via kv_cache.unbind(1).
        self.kv_caches = []
        for _ in range(self.num_layers):
            kv = torch.zeros(
                self.num_kv_blocks,
                2,
                self.block_size,
                self.num_kv_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            self.kv_caches.append(kv)

    def _bind_kv_cache_to_attention_layers(self) -> None:
        """Bind allocated KV tensors to the model's attention layers."""
        forward_ctx = (
            self._draft_vllm_config.compilation_config.static_forward_context
        )
        if not forward_ctx:
            logger.warning(
                "No attention layers found in static_forward_context. "
                "KV cache will not be bound."
            )
            return

        from vllm.model_executor.layers.attention import Attention
        from vllm.model_executor.models.utils import extract_layer_index

        attn_layers = []
        for layer_name, layer in forward_ctx.items():
            if isinstance(layer, Attention):
                try:
                    idx = extract_layer_index(layer_name)
                except (ValueError, IndexError):
                    idx = len(attn_layers)
                attn_layers.append((idx, layer_name, layer))

        attn_layers.sort(key=lambda x: x[0])

        assert self.kv_caches is not None
        if len(attn_layers) != len(self.kv_caches):
            logger.warning(
                "Mismatch: %d attention layers vs %d KV cache tensors.",
                len(attn_layers), len(self.kv_caches),
            )

        num_bind = min(len(attn_layers), len(self.kv_caches))
        for i in range(num_bind):
            _, _layer_name, layer = attn_layers[i]
            layer.kv_cache = self.kv_caches[i]

        logger.info("Bound KV cache to %d attention layers.", num_bind)

    # ----- Block allocator -----

    def _alloc_one_block(self) -> int:
        """Return a single free block ID, preferring the free list."""
        if self._free_list:
            return self._free_list.pop()
        if self._next_free_block >= self.num_kv_blocks:
            raise RuntimeError(
                f"Draft KV cache exhausted: {self._next_free_block} >= "
                f"{self.num_kv_blocks} (free_list empty)"
            )
        blk = self._next_free_block
        self._next_free_block += 1
        return blk

    def allocate_blocks(self, seq_id: int, num_tokens: int) -> list[int]:
        """Allocate KV blocks for a sequence, recycling any prior ones."""
        if seq_id >= self._block_table_gpu.shape[0]:
            raise ValueError(
                f"seq_id {seq_id} exceeds GPU block table capacity "
                f"({self._block_table_gpu.shape[0]}). Increase max_num_seqs."
            )
        if seq_id in self._block_tables:
            self._free_list.extend(self._block_tables.pop(seq_id))

        num_blocks_needed = (
            (num_tokens + self.block_size - 1) // self.block_size
        )
        blocks = [
            self._alloc_one_block() for _ in range(num_blocks_needed)
        ]
        self._block_tables[seq_id] = blocks
        self._block_table_gpu[seq_id].zero_()
        self._block_table_gpu[seq_id, :len(blocks)] = torch.tensor(
            blocks, dtype=torch.int32, device=self.device
        )
        return blocks

    def free_blocks(self, seq_id: int) -> None:
        """Free KV blocks for a completed sequence."""
        old_blocks = self._block_tables.pop(seq_id, None)
        if old_blocks:
            self._free_list.extend(old_blocks)
        self._seq_lens.pop(seq_id, None)
        if seq_id < self._block_table_gpu.shape[0]:
            self._block_table_gpu[seq_id].zero_()

    def ensure_blocks(self, seq_id: int, num_tokens: int) -> None:
        """Grow block allocation for a sequence if it needs more rows."""
        if seq_id not in self._block_tables:
            self.allocate_blocks(seq_id, num_tokens)
            return

        current_blocks = len(self._block_tables[seq_id])
        needed_blocks = (
            (num_tokens + self.block_size - 1) // self.block_size
        )
        if needed_blocks <= current_blocks:
            return

        extra = needed_blocks - current_blocks
        new_blocks = [self._alloc_one_block() for _ in range(extra)]
        self._block_tables[seq_id].extend(new_blocks)
        start = current_blocks
        self._block_table_gpu[seq_id, start:start + extra] = torch.tensor(
            new_blocks, dtype=torch.int32, device=self.device
        )

    # ----- Branch-table swap (cache-hit path) -----

    def swap_block_tables(
        self,
        seq_ids: torch.Tensor,
        branch_block_tables: torch.Tensor,
        prefix_lens: torch.Tensor,
        K: int,
    ) -> tuple[dict[int, list[int]], list[int]]:
        """Swap branch block-table entries into the main block table.

        Overwrites only the write-range columns (logical blocks the
        branch's tree decode wrote into). Returns (owned_blocks_by_sid,
        displaced) so the caller can recycle displaced blocks.
        """
        bs = self.block_size
        M = self.max_num_blocks
        B = seq_ids.shape[0]
        owned_blocks: dict[int, list[int]] = {}
        displaced: list[int] = []

        for i in range(B):
            sid = int(seq_ids[i].item())
            if sid >= self._block_table_gpu.shape[0]:
                logger.warning(
                    "swap_block_tables: seq_id %d out of bounds, skipping",
                    sid,
                )
                continue

            prefix_len = int(prefix_lens[i].item())
            first_write_blk = prefix_len // bs
            last_write_blk = (prefix_len + K - 1) // bs

            owned = []
            for blk_idx in range(
                first_write_blk, min(last_write_blk + 1, M)
            ):
                if (sid in self._block_tables
                        and blk_idx < len(self._block_tables[sid])):
                    old_blk = self._block_tables[sid][blk_idx]
                    if old_blk != 0:
                        displaced.append(old_blk)

                new_block_id = int(branch_block_tables[i, blk_idx].item())
                self._block_table_gpu[sid, blk_idx] = new_block_id
                owned.append(new_block_id)

                if sid in self._block_tables:
                    while len(self._block_tables[sid]) <= blk_idx:
                        self._block_tables[sid].append(0)
                    self._block_tables[sid][blk_idx] = new_block_id

            owned_blocks[sid] = owned

        return owned_blocks, displaced

    def release_owned_blocks(
        self, seq_id: int, owned_blocks: list[int]
    ) -> None:
        """Recycle block IDs that were swapped into main."""
        if owned_blocks:
            self._free_list.extend(owned_blocks)

    # ----- Dedicated blocks per verify server -----

    def recycle_dedicated_blocks(
        self, vs_id: str = "__default__"
    ) -> None:
        """Recycle dedicated tree-decode blocks for one verify server."""
        blocks = self._dedicated_blocks_by_vs.pop(vs_id, None)
        if blocks:
            self._free_list.extend(blocks)
        self._try_compact()

    def recycle_all_dedicated_blocks(self) -> None:
        """Recycle every VS's dedicated blocks (shutdown/testing only)."""
        for blocks in self._dedicated_blocks_by_vs.values():
            self._free_list.extend(blocks)
        self._dedicated_blocks_by_vs.clear()
        self._try_compact()

    def _try_compact(self) -> None:
        """Rewind the bump pointer by draining the top of the free list."""
        if not self._free_list:
            return
        free_set = set(self._free_list)
        while (self._next_free_block > 0
               and (self._next_free_block - 1) in free_set):
            self._next_free_block -= 1
            free_set.discard(self._next_free_block)
        self._free_list = list(free_set)

    def reserve_dedicated_blocks(
        self, block_ids: list[int], vs_id: str = "__default__"
    ) -> None:
        """Track dedicated blocks allocated for tree decode, per VS."""
        self._dedicated_blocks_by_vs[vs_id] = block_ids

    def exclude_from_dedicated(
        self, owned_blocks: list[int], vs_id: str = "__default__"
    ) -> None:
        """Remove swapped blocks from one VS's dedicated list.

        When a cache-hit branch's blocks are swapped into main they
        become owned by the sequence; dropping them from the dedicated
        list avoids double-free on the next recycle.
        """
        if not owned_blocks:
            return
        blocks = self._dedicated_blocks_by_vs.get(vs_id)
        if not blocks:
            return
        owned_set = set(owned_blocks)
        self._dedicated_blocks_by_vs[vs_id] = [
            b for b in blocks if b not in owned_set
        ]

    # ----- Snapshot / rollback (tree decode) -----

    def save_kv_snapshot(self, seq_ids: list[int]) -> None:
        """Snapshot sequence lengths so tree branching can roll back."""
        self._kv_snapshot = {
            sid: self._seq_lens.get(sid, 0) for sid in seq_ids
        }

    def rollback_kv(self, seq_ids: list[int]) -> None:
        """Restore sequence lengths from the last snapshot."""
        if self._kv_snapshot is None:
            return
        for sid in seq_ids:
            if sid in self._kv_snapshot:
                self._seq_lens[sid] = self._kv_snapshot[sid]
