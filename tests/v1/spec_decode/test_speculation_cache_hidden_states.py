# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SpeculationCache hidden states support (task 6.1).

Validates Requirements 6.1, 6.2, 6.3: the cache stores and returns
hidden states alongside draft tokens when needs_hidden_states=True.
"""

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.disagg_draft.speculation_cache import (
    SpeculationCache,
)

DEVICE = "cpu"
DTYPE = torch.bfloat16
K = 5
VOCAB = 100
FAN_OUT = 3
BATCH = 4
HIDDEN_SIZE = 64


@pytest.fixture
def cache_no_hs():
    """Cache without hidden states (standalone draft model)."""
    return SpeculationCache(
        max_batch_size=BATCH,
        num_speculative_tokens=K,
        fan_out=FAN_OUT,
        vocab_size=VOCAB,
        device=torch.device(DEVICE),
        dtype=DTYPE,
    )


@pytest.fixture
def cache_with_hs():
    """Cache with hidden states (EAGLE/EAGLE3/MTP)."""
    return SpeculationCache(
        max_batch_size=BATCH,
        num_speculative_tokens=K,
        fan_out=FAN_OUT,
        vocab_size=VOCAB,
        device=torch.device(DEVICE),
        dtype=DTYPE,
        needs_hidden_states=True,
        hidden_size=HIDDEN_SIZE,
    )


class TestInit:
    def test_no_hidden_states_by_default(self, cache_no_hs):
        assert cache_no_hs._hidden_states is None
        assert cache_no_hs.needs_hidden_states is False

    def test_hidden_states_allocated(self, cache_with_hs):
        assert cache_with_hs._hidden_states is not None
        assert cache_with_hs._hidden_states.shape == (
            cache_with_hs.max_entries,
            HIDDEN_SIZE,
        )
        assert cache_with_hs._hidden_states.dtype == DTYPE
        assert cache_with_hs.needs_hidden_states is True
        assert cache_with_hs.hidden_size == HIDDEN_SIZE

    def test_hidden_states_requires_positive_hidden_size(self):
        with pytest.raises(ValueError, match="hidden_size must be positive"):
            SpeculationCache(
                max_batch_size=BATCH,
                num_speculative_tokens=K,
                fan_out=FAN_OUT,
                vocab_size=VOCAB,
                device=torch.device(DEVICE),
                dtype=DTYPE,
                needs_hidden_states=True,
                hidden_size=0,
            )


class TestPopulate:
    def test_populate_with_hidden_states(self, cache_with_hs):
        N = 3
        seq_ids = torch.tensor([1, 2, 3], dtype=torch.int64)
        k_pos = torch.tensor([0, 1, 2], dtype=torch.int64)
        bonus = torch.tensor([10, 20, 30], dtype=torch.int64)
        tokens = torch.randint(0, VOCAB, (N, K), dtype=torch.int64)
        logits = torch.randn(N, K, VOCAB, dtype=DTYPE)
        hs = torch.randn(N, HIDDEN_SIZE, dtype=DTYPE)

        cache_with_hs.populate(
            seq_ids=seq_ids,
            k_positions=k_pos,
            bonus_tokens=bonus,
            draft_tokens=tokens,
            draft_logits=logits,
            hidden_states=hs,
        )

        assert cache_with_hs.num_entries == N
        assert torch.equal(cache_with_hs._hidden_states[:N], hs)

    def test_populate_without_hidden_states_on_hs_cache(self, cache_with_hs):
        """Populating without hidden_states when cache supports them is OK."""
        N = 2
        seq_ids = torch.tensor([1, 2], dtype=torch.int64)
        k_pos = torch.tensor([0, 1], dtype=torch.int64)
        bonus = torch.tensor([10, 20], dtype=torch.int64)
        tokens = torch.randint(0, VOCAB, (N, K), dtype=torch.int64)
        logits = torch.randn(N, K, VOCAB, dtype=DTYPE)

        # Should not raise — hidden_states is optional
        cache_with_hs.populate(
            seq_ids=seq_ids,
            k_positions=k_pos,
            bonus_tokens=bonus,
            draft_tokens=tokens,
            draft_logits=logits,
        )
        assert cache_with_hs.num_entries == N

    def test_populate_ignores_hidden_states_on_no_hs_cache(self, cache_no_hs):
        """Passing hidden_states to a cache that doesn't need them is OK."""
        N = 2
        seq_ids = torch.tensor([1, 2], dtype=torch.int64)
        k_pos = torch.tensor([0, 1], dtype=torch.int64)
        bonus = torch.tensor([10, 20], dtype=torch.int64)
        tokens = torch.randint(0, VOCAB, (N, K), dtype=torch.int64)
        logits = torch.randn(N, K, VOCAB, dtype=DTYPE)
        hs = torch.randn(N, HIDDEN_SIZE, dtype=DTYPE)

        # Should not raise — just ignored
        cache_no_hs.populate(
            seq_ids=seq_ids,
            k_positions=k_pos,
            bonus_tokens=bonus,
            draft_tokens=tokens,
            draft_logits=logits,
            hidden_states=hs,
        )
        assert cache_no_hs.num_entries == N


class TestLookup:
    def test_lookup_returns_4_tuple(self, cache_no_hs):
        """lookup() now returns a 4-tuple even without hidden states."""
        seq_ids = torch.tensor([1], dtype=torch.int64)
        k_acc = torch.tensor([0], dtype=torch.int64)
        bonus = torch.tensor([10], dtype=torch.int64)

        result = cache_no_hs.lookup(seq_ids, k_acc, bonus)
        assert len(result) == 4
        tokens, logits, hits, hs = result
        assert hs is None

    def test_empty_cache_returns_none_hidden_states(self, cache_with_hs):
        seq_ids = torch.tensor([1], dtype=torch.int64)
        k_acc = torch.tensor([0], dtype=torch.int64)
        bonus = torch.tensor([10], dtype=torch.int64)

        tokens, logits, hits, hs = cache_with_hs.lookup(
            seq_ids, k_acc, bonus
        )
        assert hits.sum() == 0
        assert hs is None

    def test_cache_hit_returns_hidden_states(self, cache_with_hs):
        """On cache hit, hidden states should be returned."""
        N = 2
        seq_ids = torch.tensor([1, 2], dtype=torch.int64)
        k_pos = torch.tensor([3, 4], dtype=torch.int64)
        bonus = torch.tensor([10, 20], dtype=torch.int64)
        tokens = torch.randint(0, VOCAB, (N, K), dtype=torch.int64)
        logits = torch.randn(N, K, VOCAB, dtype=DTYPE)
        hs = torch.randn(N, HIDDEN_SIZE, dtype=DTYPE)

        cache_with_hs.populate(
            seq_ids=seq_ids,
            k_positions=k_pos,
            bonus_tokens=bonus,
            draft_tokens=tokens,
            draft_logits=logits,
            hidden_states=hs,
        )

        # Query for seq_id=1, k=3, bonus=10 → should hit
        q_seq = torch.tensor([1], dtype=torch.int64)
        q_k = torch.tensor([3], dtype=torch.int64)
        q_bonus = torch.tensor([10], dtype=torch.int64)

        tokens_out, logits_out, hits, hs_out = cache_with_hs.lookup(
            q_seq, q_k, q_bonus
        )
        assert hits[0].item() is True
        assert hs_out is not None
        assert hs_out.shape == (1, HIDDEN_SIZE)
        assert torch.equal(hs_out[0], hs[0])

    def test_cache_miss_returns_zero_hidden_states(self, cache_with_hs):
        """On cache miss, hidden states should be zeros for miss entries."""
        N = 1
        seq_ids = torch.tensor([1], dtype=torch.int64)
        k_pos = torch.tensor([3], dtype=torch.int64)
        bonus = torch.tensor([10], dtype=torch.int64)
        tokens = torch.randint(0, VOCAB, (N, K), dtype=torch.int64)
        logits = torch.randn(N, K, VOCAB, dtype=DTYPE)
        hs = torch.randn(N, HIDDEN_SIZE, dtype=DTYPE)

        cache_with_hs.populate(
            seq_ids=seq_ids,
            k_positions=k_pos,
            bonus_tokens=bonus,
            draft_tokens=tokens,
            draft_logits=logits,
            hidden_states=hs,
        )

        # Query with wrong bonus → miss
        q_seq = torch.tensor([1], dtype=torch.int64)
        q_k = torch.tensor([3], dtype=torch.int64)
        q_bonus = torch.tensor([99], dtype=torch.int64)

        tokens_out, logits_out, hits, hs_out = cache_with_hs.lookup(
            q_seq, q_k, q_bonus
        )
        assert hits[0].item() is False
        # No hits → hs_out should be None
        assert hs_out is None

    def test_mixed_hit_miss_hidden_states(self, cache_with_hs):
        """With a mix of hits and misses, only hit entries get hidden states."""
        N = 2
        seq_ids = torch.tensor([1, 2], dtype=torch.int64)
        k_pos = torch.tensor([3, 4], dtype=torch.int64)
        bonus = torch.tensor([10, 20], dtype=torch.int64)
        tokens = torch.randint(0, VOCAB, (N, K), dtype=torch.int64)
        logits = torch.randn(N, K, VOCAB, dtype=DTYPE)
        hs = torch.randn(N, HIDDEN_SIZE, dtype=DTYPE)

        cache_with_hs.populate(
            seq_ids=seq_ids,
            k_positions=k_pos,
            bonus_tokens=bonus,
            draft_tokens=tokens,
            draft_logits=logits,
            hidden_states=hs,
        )

        # Query: seq_id=1 k=3 bonus=10 → hit, seq_id=2 k=4 bonus=99 → miss
        q_seq = torch.tensor([1, 2], dtype=torch.int64)
        q_k = torch.tensor([3, 4], dtype=torch.int64)
        q_bonus = torch.tensor([10, 99], dtype=torch.int64)

        tokens_out, logits_out, hits, hs_out = cache_with_hs.lookup(
            q_seq, q_k, q_bonus
        )
        assert hits[0].item() is True
        assert hits[1].item() is False
        assert hs_out is not None
        assert hs_out.shape == (2, HIDDEN_SIZE)
        # Hit entry should match
        assert torch.equal(hs_out[0], hs[0])
        # Miss entry should be zeros
        assert torch.equal(
            hs_out[1], torch.zeros(HIDDEN_SIZE, dtype=DTYPE)
        )

    def test_no_hs_cache_lookup_returns_none(self, cache_no_hs):
        """Cache without hidden states always returns None for hs."""
        N = 1
        seq_ids = torch.tensor([1], dtype=torch.int64)
        k_pos = torch.tensor([3], dtype=torch.int64)
        bonus = torch.tensor([10], dtype=torch.int64)
        tokens = torch.randint(0, VOCAB, (N, K), dtype=torch.int64)
        logits = torch.randn(N, K, VOCAB, dtype=DTYPE)

        cache_no_hs.populate(
            seq_ids=seq_ids,
            k_positions=k_pos,
            bonus_tokens=bonus,
            draft_tokens=tokens,
            draft_logits=logits,
        )

        q_seq = torch.tensor([1], dtype=torch.int64)
        q_k = torch.tensor([3], dtype=torch.int64)
        q_bonus = torch.tensor([10], dtype=torch.int64)

        tokens_out, logits_out, hits, hs_out = cache_no_hs.lookup(
            q_seq, q_k, q_bonus
        )
        assert hits[0].item() is True
        assert hs_out is None
