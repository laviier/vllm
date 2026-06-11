# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tree-decode and parallel-fanout strategies for cache_build.

Three methods that produce the K cached draft tokens per branch:

- ``_run_tree_decode``: K sequential decoder forwards, each fed the
  previous step's argmax. AR-style drafting; works on any drafter.
- ``_run_parallel_fanout``: one batched forward over N*K tokens with
  a mask-token at depths 1..K-1. Requires a drafter trained with the
  parallel prediction (MTP) objective.
- ``_parallel_fanout_kv_cleanup``: a follow-up parallel forward that
  overwrites the mask-derived KVs at slots prefix+1..prefix+K-1 with
  KVs from real-token embeddings, and returns the cleaner logits at
  those positions. Required after ``_run_parallel_fanout`` because the
  trained ``mask_embedding`` tends to have a much smaller norm than
  typical token embeddings; without cleanup the near-zero KVs become
  part of the seq's history (via ``swap_block_tables``) and contaminate
  next round's attention.
"""

from __future__ import annotations

from typing import Any

import torch


class DraftServerFanoutMixin:
    """Mixin: per-branch draft-token strategies (tree decode + parallel)."""

    def _run_tree_decode(
        self,
        runner: Any,
        N: int,
        K: int,
        seq_ids: torch.Tensor,
        entry_batch_ids: torch.Tensor,
        prefix_lens: torch.Tensor,
        branch_block_tables: torch.Tensor,
        bonus_candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run K tree-decode steps and return (tokens, logits) per branch."""
        seq_ids_expanded = seq_ids[entry_batch_ids]
        all_tokens = torch.zeros(
            N, K, dtype=torch.int64, device=self.device
        )
        all_logits = torch.zeros(
            N, K, self.vocab_size, dtype=self.dtype, device=self.device
        )
        current_ids = bonus_candidates.clone()
        max_context_hint = int(prefix_lens.max().item()) + K + 1

        for depth in range(K):
            tree_positions = prefix_lens + depth
            context_lens = prefix_lens + depth + 1
            logits = runner.tree_decode_step(
                input_ids=current_ids,
                positions=tree_positions,
                seq_lens=context_lens,
                seq_ids_expanded=seq_ids_expanded,
                block_tables=branch_block_tables,
                max_seq_len_hint=max_context_hint,
            )
            all_logits[:, depth] = logits
            next_tokens = logits.argmax(dim=-1)
            all_tokens[:, depth] = next_tokens
            current_ids = next_tokens

        return all_tokens, all_logits

    def _run_parallel_fanout(
        self,
        runner: Any,
        N: int,
        K: int,
        seq_ids: torch.Tensor,
        entry_batch_ids: torch.Tensor,
        prefix_lens: torch.Tensor,
        branch_block_tables: torch.Tensor,
        bonus_candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-pass parallel fanout for MTP-style draft models.

        Instead of K sequential tree_decode_step calls, generates all
        N×K tokens in ONE forward pass. The parallel draft model uses:
        - Depth-1: bonus candidate token embedding (seeds the branch)
        - Depth-2+: MTP mask token embedding (model predicts independently)

        Each token is a separate 1-token "sequence" in the varlen batch.
        Depths within a branch do NOT attend to each other (no intra-branch
        KV dependency) — they only attend to the shared prefix context.
        This is the key property of the parallel draft model that enables
        single-pass generation.

        Args:
            runner: DraftModelRunner instance
            N: number of branches
            K: speculation depth per branch
            seq_ids: [B] sequence IDs
            entry_batch_ids: [N] maps each branch to its batch index
            prefix_lens: [N] prefix length per branch (prefix + spec[:k_j])
            branch_block_tables: [N, max_blocks] per-branch block tables
            bonus_candidates: [N] seed tokens for depth-1

        Returns:
            all_tokens: [N, K] generated draft tokens
            all_logits: [N, K, V] logits at each position
        """
        total_tokens = N * K

        # --- Build input_ids: [N*K] ---
        # Layout: [br0_d0, br0_d1, ..., br0_dK-1, br1_d0, ..., brN_dK-1]
        # Depth 0 (first in each branch): bonus candidate token
        # Depth 1+ (rest): MTP mask token
        input_ids = torch.full(
            (total_tokens,), self._mtp_token_id,
            dtype=torch.int32, device=self.device,
        )
        # Set depth-0 positions to bonus candidates
        depth0_indices = torch.arange(
            0, total_tokens, K, device=self.device
        )
        input_ids[depth0_indices] = bonus_candidates.to(torch.int32)

        # --- Build positions: [N*K] ---
        # Branch j, depth d → prefix_lens[j] + d
        # Each branch starts at its prefix_len (which already includes
        # the verified prefix + accepted spec tokens up to k_j)
        positions = torch.zeros(
            total_tokens, dtype=torch.int64, device=self.device
        )
        depth_offsets = torch.arange(K, device=self.device, dtype=torch.int64)
        for branch_idx in range(N):
            start = branch_idx * K
            positions[start:start + K] = prefix_lens[branch_idx] + depth_offsets

        # --- Build seq_lens: [N*K] ---
        # Causal MTP: depth-d in a branch attends to the prefix PLUS the
        # earlier depths' K/V within the same branch. seq_len for depth-d
        # is prefix_lens[branch] + d + 1 (prefix + d earlier mask-token
        # K/V slots + the current position itself). This matches how the
        # parallel-draft model was trained (mtp_attention: causal): the
        # depth-d head expects to attend to depths 0..d-1's mask-token
        # K/V projections, not just the prefix.
        #
        # Earlier code used prefix_lens + 1 for all depths (non-causal),
        # which hid earlier depths from later ones. With a causal-trained
        # model that produced a per-position acceptance cliff (P0 70% →
        # P2 22% → P3 17%) because depths 2+ were running blind.
        seq_lens = torch.zeros(
            total_tokens, dtype=torch.int32, device=self.device
        )
        for branch_idx in range(N):
            start = branch_idx * K
            seq_lens[start:start + K] = (
                prefix_lens[branch_idx] + 1 + depth_offsets
            ).to(torch.int32)

        # --- Build block_tables: [N*K, max_blocks] ---
        # All depths in a branch share the same block table (they all
        # attend to the same prefix KV, no branch-local KV needed).
        block_tables_expanded = branch_block_tables.repeat_interleave(
            K, dim=0
        )

        # --- Run single forward pass ---
        max_context_hint = int(prefix_lens.max().item()) + K + 1
        logits_flat = runner.tree_decode_step(
            input_ids=input_ids,
            positions=positions,
            seq_lens=seq_lens,
            seq_ids_expanded=seq_ids[entry_batch_ids].repeat_interleave(K),
            block_tables=block_tables_expanded,
            max_seq_len_hint=max_context_hint,
        )

        # --- Reshape outputs: [N*K] → [N, K] ---
        all_logits = logits_flat.view(N, K, -1)
        all_tokens = all_logits.argmax(dim=-1)

        return all_tokens, all_logits

    def _parallel_fanout_kv_cleanup(
        self,
        runner: Any,
        N: int,
        K: int,
        seq_ids: torch.Tensor,
        entry_batch_ids: torch.Tensor,
        prefix_lens: torch.Tensor,
        branch_block_tables: torch.Tensor,
        all_tokens: torch.Tensor,
    ) -> torch.Tensor | None:
        """Overwrite mask-derived KVs at positions prefix+1..prefix+K-1
        with KVs from real-token embeddings.

        After parallel fanout, slots ``prefix..prefix+K-1`` of each
        branch's dedicated block hold KVs computed from the inputs
        ``[bonus_candidate, mask, mask, ..., mask]``. The mask-token
        embedding has near-zero norm in this checkpoint, so positions
        ``prefix+1..prefix+K-1`` contain near-zero KVs that contaminate
        future rounds (after the branch's block becomes the seq's main
        block via ``swap_hits``).

        This cleanup re-feeds the model with parallel-fanout's
        argmax-predicted tokens at those positions, producing real-
        token-derived KVs that overwrite the mask-derived ones at the
        same slots. Logits from this cleanup are discarded; we keep
        ``all_tokens`` as the parallel-fanout result.

        Implementation: one parallel forward over N*(K-1) tokens. Each
        depth d at position prefix+d attends causally to the prefix
        plus all earlier depths' KV (which the layer's K/V write phase
        emits before attention reads, so depth d sees real-token KVs
        from depths 1..d-1 in the same forward). Equivalent to running
        K-1 sequential forwards but ~K-1× faster.
        """
        if K <= 1:
            return None
        D = K - 1
        total_tokens = N * D

        # input_ids: [N*D] — for branch j depth d (1-indexed 1..K-1),
        # input is parallel-fanout's predicted token at depth d-1.
        # Layout: [br0_d1, br0_d2, ..., br0_d{K-1}, br1_d1, ..., brN_d{K-1}]
        input_ids = all_tokens[:, :D].to(torch.int32).reshape(total_tokens)

        # positions: [N*D] — branch j depth d (1..K-1) at prefix_lens[j] + d
        depth_offsets = torch.arange(
            1, K, device=self.device, dtype=torch.int64,
        )  # [1, 2, ..., K-1] of length D
        positions = (
            prefix_lens.unsqueeze(1) + depth_offsets.unsqueeze(0)
        ).reshape(total_tokens).to(torch.int64)

        # seq_lens: [N*D] — depth d attends causally to prefix +
        # earlier depths' (real-token, just-cleaned) KVs in the same
        # forward. seq_len = prefix_lens[j] + d + 1.
        seq_lens = (
            prefix_lens.unsqueeze(1) + 1 + depth_offsets.unsqueeze(0)
        ).reshape(total_tokens).to(torch.int32)

        # block_tables: [N*D, max_blocks] — all D depths in a branch
        # share the same dedicated block table.
        block_tables_expanded = branch_block_tables.repeat_interleave(
            D, dim=0,
        )

        seq_ids_for_branches = seq_ids[entry_batch_ids]
        seq_ids_expanded = seq_ids_for_branches.repeat_interleave(D)

        max_context_hint = int(prefix_lens.max().item()) + K
        # Capture logits — they predict positions prefix+2..prefix+K
        # using real-token KVs at all earlier depths, so they're cleaner
        # than parallel-fanout's mask-token-derived logits at the same
        # positions. Caller may use them to overwrite all_logits[:, 1:].
        cleanup_logits_flat = runner.tree_decode_step(
            input_ids=input_ids,
            positions=positions,
            seq_lens=seq_lens,
            seq_ids_expanded=seq_ids_expanded,
            block_tables=block_tables_expanded,
            max_seq_len_hint=max_context_hint,
        )
        # Reshape [N*D, V] -> [N, D, V]
        return cleanup_logits_flat.view(N, D, -1)

