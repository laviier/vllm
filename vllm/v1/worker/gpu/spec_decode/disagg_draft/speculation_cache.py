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

        # Pinned CPU staging + GPU staging for ``get_hit_block_tables``.
        # Used to avoid ``owners.tolist() + hit_indices.tolist()`` host
        # syncs, which stall behind cache_build kernels on the default
        # stream when the SPECULATE handler is called concurrently.
        self._vs_start_cpu_pin = torch.zeros(
            max_verify_servers, dtype=torch.int64, pin_memory=True,
        )
        self._vs_offset_cpu_pin = torch.zeros(
            max_verify_servers, dtype=torch.int64, pin_memory=True,
        )
        self._vs_start_gpu = torch.zeros(
            max_verify_servers, dtype=torch.int64, device=device,
        )
        self._vs_offset_gpu = torch.zeros(
            max_verify_servers, dtype=torch.int64, device=device,
        )

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

    def drop_entries_by_seq_ids(
        self, vs_id: str, seq_ids_to_drop: list[int] | set[int]
    ) -> None:
        """Remove only the cache entries whose seq_id is in the given set,
        leaving the rest of the VS partition intact.

        Used from ``_handle_free_seq`` on the draft server. Wiping the
        whole VS partition (``reset_vs``) on every FREE_SEQ invalidates
        entries for *other* active sequences belonging to the same VS,
        which collapses cache hit rate under high-concurrency 2V+1D
        workloads. This variant only drops entries for the freed sids,
        preserving entries for sequences that are still live.

        Correctness: the cached entries for sids that are NOT in the
        drop set still point at dedicated blocks reserved under
        ``_dedicated_blocks_by_vs[vs_id]``, which remain live because
        we don't call ``recycle_dedicated_blocks(vs_id)`` here. Any
        future cache hit on a kept entry will still find valid KV data
        via ``swap_block_tables``.

        The freed sids, meanwhile, may be recycled by the shared
        ``_free_internal_seq_ids`` pool and reassigned to another VS
        on its next PREFILL. Removing their entries here is the
        stale-sid protection that prevents a peer VS from hitting on
        pre-free data.
        """
        offset = self._vs_offsets.get(vs_id)
        count = self._vs_counts.get(vs_id, 0)
        if offset is None or count == 0 or not seq_ids_to_drop:
            return

        # Build a boolean mask [count] over this partition's slice.
        drop_set = torch.tensor(
            list(seq_ids_to_drop),
            dtype=torch.int64,
            device=self.device,
        )
        partition_seq_ids = self.keys[offset : offset + count, 0]
        # [count] True where the entry's seq_id is in the drop set.
        drop_mask = torch.isin(partition_seq_ids, drop_set)
        n_drop = int(drop_mask.sum().item())
        if n_drop == 0:
            return
        if n_drop == count:
            # Entire partition was dropped — fall back to reset_vs.
            self.reset_vs(vs_id)
            return

        keep_mask = ~drop_mask
        keep_count = count - n_drop

        # Gather kept rows within this partition.
        keep_idx = keep_mask.nonzero(as_tuple=True)[0]           # [keep_count]
        # Global indices of kept rows:
        kept_global = keep_idx + offset                           # [keep_count]

        # Overwrite the partition's slice with kept rows compacted
        # to the front. We use gather via fancy indexing (safe: dest
        # and source don't alias because we write back into the same
        # slice starting at ``offset``, but compacted).
        kept_keys = self.keys[kept_global].clone()
        self.keys[offset : offset + keep_count] = kept_keys

        kept_tokens = self.tokens[kept_global].clone()
        self.tokens[offset : offset + keep_count] = kept_tokens

        kept_owners = self._owners[kept_global].clone()
        self._owners[offset : offset + keep_count] = kept_owners

        if self._logits is not None and self._logits_allocated >= offset + count:
            kept_logits = self._logits[kept_global].clone()
            self._logits[offset : offset + keep_count] = kept_logits

        if self._hidden_states is not None:
            kept_hs = self._hidden_states[kept_global].clone()
            self._hidden_states[offset : offset + keep_count] = kept_hs

        # Update the per-VS branch_block_tables / prefix_lens to match
        # the new partition layout. These are kept as dense tensors
        # indexed 0..partition_count-1.
        tbl = self._vs_branch_block_tables.get(vs_id)
        if tbl is not None:
            self._vs_branch_block_tables[vs_id] = tbl[keep_mask].contiguous()
        plen = self._vs_prefix_lens.get(vs_id)
        if plen is not None:
            self._vs_prefix_lens[vs_id] = plen[keep_mask].contiguous()

        # Close the gap left between the end of this partition's new
        # valid range and the start of the next partition by shifting
        # later partitions down by n_drop.
        partition_end_old = offset + count
        partition_end_new = offset + keep_count
        gap = n_drop
        N = self.num_entries
        if partition_end_old < N:
            tail = N - partition_end_old
            self.keys[partition_end_new : partition_end_new + tail] = (
                self.keys[partition_end_old:N].clone()
            )
            self.tokens[partition_end_new : partition_end_new + tail] = (
                self.tokens[partition_end_old:N].clone()
            )
            self._owners[partition_end_new : partition_end_new + tail] = (
                self._owners[partition_end_old:N].clone()
            )
            if self._logits is not None and self._logits_allocated >= N:
                self._logits[partition_end_new : partition_end_new + tail] = (
                    self._logits[partition_end_old:N].clone()
                )
            if self._hidden_states is not None:
                self._hidden_states[partition_end_new : partition_end_new + tail] = (
                    self._hidden_states[partition_end_old:N].clone()
                )
            # Shift down offsets for later partitions.
            for other_vs, other_off in list(self._vs_offsets.items()):
                if other_off >= partition_end_old:
                    self._vs_offsets[other_vs] = other_off - gap

        self._vs_counts[vs_id] = keep_count
        self.num_entries = N - gap

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

        # NOTE: skip cache_hits.sum().item() here — it forces a CPU sync
        # on the default stream, which in the IPC-early-dispatch path
        # stalls this handler behind cache_build kernels from the prior
        # round. Metrics are accumulated on the caller side via
        # ``_accumulate_hit_metrics``, which lazily .item()-s a GPU-side
        # running sum every N rounds instead.

        # Extract matched entries WITHOUT boolean-mask indexing.
        # Boolean-mask indexing (``x[bool_mask]``) is a synchronizing op
        # in PyTorch: the output shape depends on how many True elements
        # the mask has, so an ``_local_scalar_dense`` fires under the
        # hood. On the IPC-early-dispatch path that sync stalls ~3 ms
        # behind the prior round's cache_build kernels.
        #
        # Use ``torch.where`` + full-index gathers instead:
        #   - ``match_idx`` is safe to use even on miss rows because
        #     ``argmax`` returns 0 (or any valid index) on all-False
        #     rows; we ignore those values via the mask in ``where``.
        #   - Gather every row from ``self.tokens[match_idx]`` (shape
        #     [B, K]) unconditionally, then blend with zeros using
        #     ``where(cache_hits, gathered, zeros)``.
        match_idx = match.float().argmax(dim=1)  # [B]
        self._last_match_idx = match_idx
        hit_mask = cache_hits
        hit_mask_kt = hit_mask.unsqueeze(-1).expand(-1, self.K)  # [B, K]

        gathered_tokens = self.tokens[match_idx]  # [B, K]
        draft_tokens_out = torch.where(
            hit_mask_kt,
            gathered_tokens,
            torch.zeros_like(gathered_tokens),
        )

        draft_logits_out: torch.Tensor | None = None
        if self._logits is not None and self._logits_allocated >= N:
            hit_mask_ktv = hit_mask.view(B, 1, 1).expand(
                -1, self.K, self.vocab_size,
            )
            gathered_logits = self._logits[match_idx]  # [B, K, V]
            draft_logits_out = torch.where(
                hit_mask_ktv,
                gathered_logits,
                torch.zeros_like(gathered_logits),
            )

        hidden_states_out: torch.Tensor | None = None
        if self._hidden_states is not None:
            hit_mask_h = hit_mask.unsqueeze(-1).expand(-1, self.hidden_size)
            gathered_hidden = self._hidden_states[match_idx]  # [B, H]
            hidden_states_out = torch.where(
                hit_mask_h,
                gathered_hidden,
                torch.zeros_like(gathered_hidden),
            )

        return draft_tokens_out, draft_logits_out, cache_hits, hidden_states_out

    def get_hit_block_tables(
        self, hit_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Get branch block tables and prefix_lens for cache hits.

        Routes each hit back to the owning VS's per-branch tables via
        ``self._owners``. All active VSes share one ``DraftModelRunner``
        so block-table column count M is identical across VSes;
        concat-and-index works safely.

        Must be called SYNCHRONOUSLY within the same round as the
        ``lookup()`` that produced ``hit_mask``. Deferring it — e.g.
        stashing hit_mask + match_idx and reading later — is unsafe:
        a peer VS's ``reset_vs`` / ``drop_entries_by_seq_ids`` between
        the lookup and this call can compact the cache and silently
        invalidate ``self._last_match_idx``.

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

        # Defensive: match_idx values come from argmax on the [B, N]
        # match tensor, so they're always in [0, N). Clamp to be safe
        # against a race where N shrank between lookup() and here.
        # Clamp uses the current num_entries; if any values would have
        # been valid but too high, they're forced into range and the
        # resulting owners row is meaningless — but the caller only
        # uses these on the hit_mask=True paths and the KV blocks it
        # writes into were reserved on the drafter side for this VS.
        n_entries = self.num_entries
        if n_entries == 0:
            return None, None
        hit_indices = hit_indices.clamp(max=n_entries - 1)

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

        # Vectorized on GPU. Reuse pre-allocated pinned+GPU staging so
        # the per-small_id lookup tensors can be updated cheaply each
        # call. Previous ``owners.tolist() + hit_indices.tolist()``
        # host syncs stalled ~3-5 ms behind cache_build kernels on the
        # IPC-early-dispatch path.
        n_small = len(self._small_id_to_vs)
        start_pin = self._vs_start_cpu_pin[:n_small]
        offset_pin = self._vs_offset_cpu_pin[:n_small]
        start_gpu = self._vs_start_gpu[:n_small]
        offset_gpu = self._vs_offset_gpu[:n_small]

        # Fill CPU-side pinned buffers with current per-small_id starts
        # / offsets. Cheap: n_small ≤ max_verify_servers (a few).
        start_pin.zero_()
        offset_pin.zero_()
        for small in range(n_small):
            vs_id = self._small_id_to_vs[small]
            if vs_id in vs_start:
                start_pin[small] = vs_start[vs_id]
            if vs_id in self._vs_offsets:
                offset_pin[small] = self._vs_offsets[vs_id]
        # Pinned → GPU with non_blocking; doesn't serialize with the
        # default stream because the source is pinned.
        start_gpu.copy_(start_pin, non_blocking=True)
        offset_gpu.copy_(offset_pin, non_blocking=True)

        # owners is [num_hits] of small_ids (int32). Gather per-hit
        # starts / offsets, then compute flat indices in one op.
        owners_i64 = owners.to(torch.int64)
        per_hit_start = start_gpu[owners_i64]              # [num_hits]
        per_hit_offset = offset_gpu[owners_i64]            # [num_hits]
        flat_idx = per_hit_start + (hit_indices - per_hit_offset)

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
