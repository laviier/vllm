# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Speculation Cache for disagg draft (Speculative Speculative Decoding).

The speculation cache stores pre-computed draft speculations indexed by
verification outcomes: (seq_id, k_accepted, bonus_token) → draft_tokens + logits.

During verification, the draft model pre-computes speculations for the most
likely outcomes. When verification completes, we look up the actual outcome
in the cache. On hit (~88% at T=0), the pre-computed tokens are returned
instantly with zero draft latency.

The cache is tensor-backed for GPU-resident operation with no CPU sync.

Reference: SSD paper §4.1 (Geometric Fan-Out Cache Construction)
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class SpeculationCache:
    """GPU-resident speculation cache mapping verification outcomes to
    pre-computed draft tokens and logits.

    The cache is keyed by (seq_id, k_accepted, bonus_token) tuples and
    stores corresponding draft token sequences and their logits for
    rejection sampling.

    All data lives on GPU as contiguous tensors for zero-copy lookups.
    The cache is rebuilt every speculation round (not persistent across rounds).

    Args:
        max_batch_size: Maximum number of sequences in a batch.
        num_speculative_tokens: K, the speculation lookahead depth.
        fan_out: F, number of bonus token candidates per acceptance position.
        vocab_size: Vocabulary size for logit storage.
        device: CUDA device for all cache tensors.
        dtype: Data type for logit tensors (default: bfloat16).
        needs_hidden_states: Whether to store EAGLE head output hidden states
            alongside draft tokens (for Hidden_State_Methods).
        hidden_size: Size of hidden state vectors. Required when
            needs_hidden_states is True.
    """

    def __init__(
        self,
        max_batch_size: int,
        num_speculative_tokens: int,
        fan_out: int,
        vocab_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        needs_hidden_states: bool = False,
        hidden_size: int = 0,
        max_verify_servers: int = 8,
    ):
        self.max_batch_size = max_batch_size
        self.K = num_speculative_tokens
        self.F = fan_out
        self.vocab_size = vocab_size
        self.device = device
        self.dtype = dtype
        self.needs_hidden_states = needs_hidden_states
        self.hidden_size = hidden_size
        self.max_verify_servers = max_verify_servers

        # Total cache entries per batch: B × (K+1) × F
        # Each acceptance position k ∈ [0, K] has F bonus token candidates.
        # k=0 means 0 tokens accepted (all rejected), bonus is the resampled token.
        # k=K means all K accepted, bonus is the standard bonus token.
        self.entries_per_seq = (self.K + 1) * self.F
        # Total capacity holds simultaneous full-capacity rounds from
        # every connected verify server so one VS's populate doesn't
        # evict another's entries. Per-VS partitioning is the fix for
        # 3V+2D cache thrashing (see draft_server.py / dedicated
        # blocks are also partitioned per VS in DraftModelRunner).
        self.max_entries = (
            max_batch_size * self.entries_per_seq * max_verify_servers
        )

        # Cache keys: [max_entries, 3] — (seq_id, k_accepted, bonus_token).
        # Internal seq_ids are globally unique across VSes (remapped by
        # DraftServer._map_seq_id), so seq_id alone disambiguates.
        self.keys = torch.zeros(
            self.max_entries, 3, dtype=torch.int64, device=device
        )
        self.tokens = torch.zeros(
            self.max_entries, self.K, dtype=torch.int64, device=device
        )
        # Lazy-allocated logits: [max_entries, K, vocab_size].
        self._logits: torch.Tensor | None = None
        self._logits_allocated = 0

        self._hidden_states: torch.Tensor | None = None
        if needs_hidden_states:
            if hidden_size <= 0:
                raise ValueError(
                    "hidden_size must be positive when "
                    "needs_hidden_states=True"
                )
            self._hidden_states = torch.zeros(
                self.max_entries,
                hidden_size,
                dtype=dtype,
                device=device,
            )

        # Number of valid entries currently in the cache (union across
        # all per-VS partitions).
        self.num_entries = 0

        # Per-VS partition metadata. Each VS gets a contiguous slice of
        # the cache tensors [offset : offset + count). populate(vs_id)
        # overwrites that slice; reset_vs(vs_id) compacts it out by
        # shifting later partitions down, keeping self.keys[:num_entries]
        # a dense view for the vectorized lookup kernel.
        self._vs_offsets: dict[str, int] = {}
        self._vs_counts: dict[str, int] = {}
        # Per-VS block-table + prefix-len tensors so cache hits can be
        # resolved to the right VS's dedicated-block layout.
        self._vs_branch_block_tables: dict[str, torch.Tensor] = {}
        self._vs_prefix_lens: dict[str, torch.Tensor] = {}
        # Owner tensor per cache slot: small_id that maps back to vs_id.
        self._owners = torch.zeros(
            self.max_entries, dtype=torch.int32, device=device
        )
        self._vs_small_id: dict[str, int] = {}
        self._small_id_to_vs: list[str] = []

        # Track per-round statistics
        self._total_lookups = 0
        self._total_hits = 0

    @property
    def hit_rate(self) -> float:
        """Running cache hit rate."""
        if self._total_lookups == 0:
            return 0.0
        return self._total_hits / self._total_lookups

    def reset(self) -> None:
        """Clear the entire cache across all verify servers.

        Used during shutdown / testing. Per-round code paths should
        use ``reset_vs(vs_id)`` so concurrent VSes' preserved entries
        survive a peer's cache rebuild.
        """
        self.num_entries = 0
        self._vs_offsets.clear()
        self._vs_counts.clear()
        self._vs_branch_block_tables.clear()
        self._vs_prefix_lens.clear()
        self._vs_small_id.clear()
        self._small_id_to_vs.clear()

    def _small_id_for_vs(self, vs_id: str) -> int:
        """Assign or retrieve a stable small-int id for this VS."""
        sid = self._vs_small_id.get(vs_id)
        if sid is not None:
            return sid
        sid = len(self._small_id_to_vs)
        self._vs_small_id[vs_id] = sid
        self._small_id_to_vs.append(vs_id)
        return sid

    def reset_vs(self, vs_id: str) -> None:
        """Remove this VS's entries, compacting later partitions down.

        Keeps ``self.keys[:num_entries]`` dense so the lookup kernel
        stays a single vectorized compare. O(num_entries_after) GPU
        copies; cheap at these sizes.
        """
        offset = self._vs_offsets.pop(vs_id, None)
        count = self._vs_counts.pop(vs_id, 0)
        self._vs_branch_block_tables.pop(vs_id, None)
        self._vs_prefix_lens.pop(vs_id, None)
        if offset is None or count == 0:
            return

        end = offset + count
        N = self.num_entries
        if end < N:
            tail = N - end
            self.keys[offset : offset + tail] = self.keys[end:N].clone()
            self.tokens[offset : offset + tail] = self.tokens[end:N].clone()
            self._owners[offset : offset + tail] = (
                self._owners[end:N].clone()
            )
            if self._logits is not None and self._logits_allocated >= N:
                self._logits[offset : offset + tail] = (
                    self._logits[end:N].clone()
                )
            if self._hidden_states is not None:
                self._hidden_states[offset : offset + tail] = (
                    self._hidden_states[end:N].clone()
                )
            # Shift down the offsets of partitions that moved.
            for other_vs, other_off in list(self._vs_offsets.items()):
                if other_off >= end:
                    self._vs_offsets[other_vs] = other_off - count
        self.num_entries = N - count

    def _ensure_logits(self, num_entries: int) -> torch.Tensor:
        """Lazily allocate or expand logits tensor as needed."""
        if self._logits is None or num_entries > self._logits_allocated:
            self._logits_allocated = num_entries
            self._logits = torch.zeros(
                num_entries,
                self.K,
                self.vocab_size,
                dtype=self.dtype,
                device=self.device,
            )
        return self._logits

    def populate(
        self,
        seq_ids: torch.Tensor,
        k_positions: torch.Tensor,
        bonus_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor,
        branch_block_tables: torch.Tensor | None = None,
        prefix_lens: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        vs_id: str = "__default__",
    ) -> None:
        """Populate the cache with pre-computed speculations for one VS.

        Entries for ``vs_id`` replace any prior entries for that VS
        (via an internal ``reset_vs`` call), then get appended after
        every other VS's partition. Concurrent VSes do not evict each
        other's speculations — which is the whole point of this data
        structure under multi-VS load. See also
        DraftModelRunner._dedicated_blocks_by_vs, which must be
        partitioned in lockstep for cache hits to remain safe.

        Args:
            seq_ids: [N] — sequence IDs for each entry.
            k_positions: [N] — acceptance position (0..K) for each entry.
            bonus_tokens: [N] — predicted bonus token for each entry.
            draft_tokens: [N, K] — pre-speculated draft token sequences.
            draft_logits: [N, K, V] — draft logits per speculated position.
            branch_block_tables: [N, M] — per-branch block tables for
                swapping on cache hit.
            prefix_lens: [N] — prefix length per branch.
            hidden_states: [N, hidden_size] — EAGLE head output hidden
                states. Only used when ``needs_hidden_states=True``.
            vs_id: verify server that owns these entries. Default is
                ``"__default__"`` for the 1:1 NCCL path where there is
                only ever one VS.
        """
        N = seq_ids.shape[0]

        # Drop this VS's prior entries first (keeps num_entries an
        # accurate append offset). No-op if this is the first populate
        # for this VS.
        self.reset_vs(vs_id)

        if N == 0:
            return

        offset = self.num_entries
        end = offset + N
        if end > self.max_entries:
            logger.warning(
                "SpeculationCache capacity exceeded: need %d slots "
                "but only %d available (num_entries=%d, vs_id=%s). "
                "Dropping populate.",
                N, self.max_entries - offset, self.num_entries, vs_id,
            )
            return

        self.keys[offset:end, 0] = seq_ids
        self.keys[offset:end, 1] = k_positions
        self.keys[offset:end, 2] = bonus_tokens
        self.tokens[offset:end] = draft_tokens

        logits_buf = self._ensure_logits(max(end, self._logits_allocated))
        logits_buf[offset:end] = draft_logits

        if self._hidden_states is not None and hidden_states is not None:
            self._hidden_states[offset:end] = hidden_states

        small_id = self._small_id_for_vs(vs_id)
        self._owners[offset:end] = small_id

        self._vs_offsets[vs_id] = offset
        self._vs_counts[vs_id] = N
        self._vs_branch_block_tables[vs_id] = branch_block_tables
        self._vs_prefix_lens[vs_id] = prefix_lens

        self.num_entries = end

    def lookup(
        self,
        seq_ids: torch.Tensor,
        k_accepted: torch.Tensor,
        bonus_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor,
               torch.Tensor | None]:
        """Look up verification outcomes in the cache.

        Returns pre-computed draft tokens, logits, and optionally hidden states
        for cache hits, and a boolean mask indicating which sequences hit.

        Args:
            seq_ids: [B] — sequence IDs to look up.
            k_accepted: [B] — number of tokens accepted per sequence.
            bonus_tokens: [B] — bonus token sampled per sequence.

        Returns:
            draft_tokens: [B, K] — pre-computed tokens (valid only where hit=True).
            draft_logits: [B, K, V] or None — pre-computed logits (None if no
                logits were stored).
            cache_hits: [B] — boolean mask, True where the outcome was cached.
            hidden_states: [B, hidden_size] or None — cached EAGLE head output
                hidden states (None if needs_hidden_states is False or no hits).
        """
        B = seq_ids.shape[0]
        assert k_accepted.shape == (B,)
        assert bonus_tokens.shape == (B,)

        self._total_lookups += B

        if self.num_entries == 0:
            # Empty cache — all misses
            return (
                torch.zeros(B, self.K, dtype=torch.int64, device=self.device),
                None,
                torch.zeros(B, dtype=torch.bool, device=self.device),
                None,
            )

        # Build query keys: [B, 3]
        query_keys = torch.stack([seq_ids, k_accepted, bonus_tokens], dim=1)

        # Vectorized lookup: compare each query against all cache entries
        # query_keys: [B, 1, 3], cache_keys: [1, N, 3]
        N = self.num_entries
        cache_keys = self.keys[:N]  # [N, 3]
        eq = query_keys.unsqueeze(1) == cache_keys.unsqueeze(0)  # [B, N, 3]
        match = eq.all(dim=2)  # [B, N]
        cache_hits = match.any(dim=1)  # [B]

        self._total_hits += int(cache_hits.sum().item())

        # Extract matched entries
        draft_tokens_out = torch.zeros(
            B, self.K, dtype=torch.int64, device=self.device
        )
        draft_logits_out = None
        hidden_states_out = None
        # match_idx for block table swapping (stored even if no logits)
        self._last_match_idx = None

        if cache_hits.any():
            match_idx = match.float().argmax(dim=1)  # [B]
            hit_mask = cache_hits
            self._last_match_idx = match_idx

            draft_tokens_out[hit_mask] = self.tokens[match_idx[hit_mask]]

            if self._logits is not None and self._logits_allocated >= N:
                draft_logits_out = torch.zeros(
                    B,
                    self.K,
                    self.vocab_size,
                    dtype=self.dtype,
                    device=self.device,
                )
                draft_logits_out[hit_mask] = self._logits[match_idx[hit_mask]]

            # Return cached hidden states for EAGLE/EAGLE3/MTP methods
            if self._hidden_states is not None:
                hidden_states_out = torch.zeros(
                    B,
                    self.hidden_size,
                    dtype=self.dtype,
                    device=self.device,
                )
                hidden_states_out[hit_mask] = (
                    self._hidden_states[match_idx[hit_mask]]
                )

        return draft_tokens_out, draft_logits_out, cache_hits, hidden_states_out

    def get_hit_block_tables(
        self, hit_mask: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Get branch block tables and prefix_lens for cache hits.

        Routes each hit back to the owning VS's per-branch tables via
        ``self._owners``. All active VSes share one ``DraftModelRunner``
        so block-table column count M is identical across VSes;
        concat-and-index works safely.

        Args:
            hit_mask: [B] — boolean mask from lookup().

        Returns:
            block_tables: [num_hits, M] or None — branch block tables.
            prefix_lens: [num_hits] or None — prefix lengths.
        """
        if self._last_match_idx is None:
            return None, None
        if not self._vs_branch_block_tables:
            return None, None

        match_idx = self._last_match_idx                  # [B]
        hit_indices = match_idx[hit_mask]                 # [num_hits]
        if hit_indices.numel() == 0:
            return None, None

        owners = self._owners[hit_indices]                # [num_hits]

        # Build a flat concatenation of per-VS branch tables indexed
        # by small_id. Per-hit lookup maps global cache index →
        # (owner_vs, local_index) → flat index in the concat.
        table_list: list[torch.Tensor] = []
        prefix_list: list[torch.Tensor] = []
        vs_start: dict[str, int] = {}
        cursor = 0
        # Iterate in small_id order so concat matches owner values.
        for small in range(len(self._small_id_to_vs)):
            vs_id = self._small_id_to_vs[small]
            tbl = self._vs_branch_block_tables.get(vs_id)
            plen = self._vs_prefix_lens.get(vs_id)
            if tbl is None or plen is None:
                # Placeholder (this VS's entries were reset); skip.
                continue
            table_list.append(tbl)
            prefix_list.append(plen)
            vs_start[vs_id] = cursor
            cursor += self._vs_counts.get(vs_id, 0)

        if not table_list:
            return None, None

        flat_tables = torch.cat(table_list, dim=0)
        flat_prefix = torch.cat(prefix_list, dim=0)

        # For each hit: local_offset = global_idx - vs_offsets[owner_vs],
        #               flat_idx = vs_start[owner_vs] + local_offset.
        # Compute on CPU (num_hits is small, typically <= B_active).
        owners_cpu = owners.tolist()
        hit_idx_cpu = hit_indices.tolist()
        flat_idx_list = []
        for owner_small, global_idx in zip(owners_cpu, hit_idx_cpu):
            vs_id = self._small_id_to_vs[owner_small]
            local_offset = global_idx - self._vs_offsets[vs_id]
            flat_idx_list.append(vs_start[vs_id] + local_offset)
        flat_idx = torch.tensor(
            flat_idx_list, dtype=torch.int64, device=hit_indices.device,
        )

        return flat_tables[flat_idx], flat_prefix[flat_idx]

    def get_stats(self) -> dict[str, float]:
        """Return cache statistics for logging."""
        return {
            "disagg_cache_entries": self.num_entries,
            "disagg_cache_total_lookups": self._total_lookups,
            "disagg_cache_total_hits": self._total_hits,
            "disagg_cache_hit_rate": self.hit_rate,
        }

    def reset_stats(self) -> None:
        """Reset running statistics."""
        self._total_lookups = 0
        self._total_hits = 0
