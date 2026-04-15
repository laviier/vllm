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
    ):
        self.max_batch_size = max_batch_size
        self.K = num_speculative_tokens
        self.F = fan_out
        self.vocab_size = vocab_size
        self.device = device
        self.dtype = dtype
        self.needs_hidden_states = needs_hidden_states
        self.hidden_size = hidden_size

        # Total cache entries per batch: B × (K+1) × F
        # Each acceptance position k ∈ [0, K] has F bonus token candidates.
        # k=0 means 0 tokens accepted (all rejected), bonus is the resampled token.
        # k=K means all K accepted, bonus is the standard bonus token.
        self.entries_per_seq = (self.K + 1) * self.F
        self.max_entries = max_batch_size * self.entries_per_seq

        # Cache keys: [max_entries, 3] — (seq_id, k_accepted, bonus_token)
        self.keys = torch.zeros(
            self.max_entries, 3, dtype=torch.int64, device=device
        )
        # Cache values: draft tokens [max_entries, K]
        self.tokens = torch.zeros(
            self.max_entries, self.K, dtype=torch.int64, device=device
        )
        # Cache values: draft logits [max_entries, K, vocab_size]
        # NOTE: This is the largest allocation. For K=7, V=128000, F=3, B=16:
        # 16 * 8 * 3 * 7 * 128000 * 2 bytes = ~5.2 GB in bf16.
        # We use lazy allocation to only allocate what's needed per round.
        self._logits: torch.Tensor | None = None
        self._logits_allocated = 0

        # Optional hidden states for EAGLE/EAGLE3/MTP methods.
        # Stores the EAGLE head's output hidden states per cache entry,
        # enabling hidden state restoration on cache hit.
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

        # Number of valid entries currently in the cache
        self.num_entries = 0

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
        """Clear the cache for a new speculation round.

        This is called at the start of each draft step. We don't zero
        the tensors — just reset the entry count. Old data is overwritten
        when new entries are populated.
        """
        self.num_entries = 0

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
    ) -> None:
        """Populate the cache with pre-computed speculations.

        Args:
            seq_ids: [N] — sequence IDs for each entry.
            k_positions: [N] — acceptance position (0..K) for each entry.
            bonus_tokens: [N] — predicted bonus token for each entry.
            draft_tokens: [N, K] — pre-speculated draft token sequences.
            draft_logits: [N, K, V] — draft logits for each speculated position.
            branch_block_tables: [N, M] — per-branch block tables for swapping.
            prefix_lens: [N] — prefix length for each branch (for _seq_lens update).
            hidden_states: [N, hidden_size] — EAGLE head output hidden states
                per entry. Only used when needs_hidden_states=True.
        """
        N = seq_ids.shape[0]
        assert N <= self.max_entries, (
            f"Too many cache entries: {N} > {self.max_entries}"
        )

        self.keys[:N, 0] = seq_ids
        self.keys[:N, 1] = k_positions
        self.keys[:N, 2] = bonus_tokens
        self.tokens[:N] = draft_tokens

        logits_buf = self._ensure_logits(N)
        logits_buf[:N] = draft_logits

        # Store hidden states for EAGLE/EAGLE3/MTP methods
        if self._hidden_states is not None and hidden_states is not None:
            self._hidden_states[:N] = hidden_states

        # Store block tables for block table swapping on cache hit
        self._branch_block_tables = branch_block_tables
        self._prefix_lens = prefix_lens

        self.num_entries = N

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

        Call after lookup() to retrieve the block tables for hit entries.
        Used for block table swapping — the hit branch's block table
        replaces the main sequence's block table.

        Args:
            hit_mask: [B] — boolean mask from lookup().

        Returns:
            block_tables: [num_hits, M] or None — branch block tables for hits.
            prefix_lens: [num_hits] or None — prefix lengths for hits.
        """
        if (self._last_match_idx is None
                or self._branch_block_tables is None
                or self._prefix_lens is None):
            return None, None

        match_idx = self._last_match_idx
        hit_indices = match_idx[hit_mask]
        return (
            self._branch_block_tables[hit_indices],
            self._prefix_lens[hit_indices],
        )

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
