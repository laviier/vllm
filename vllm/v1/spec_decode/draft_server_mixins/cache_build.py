# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache-build orchestration for ``DraftServer``.

Builds the speculation cache for the next SPECULATE round so the
verifier can swap pre-computed drafts in via ``cache.lookup``
instead of waiting on a JIT drafter forward. Runs as an asyncio
background task that the next round's handler awaits before
mutating shared state.

Expects the consumer to expose: ``draft_model_runner``, ``cache``,
``outcome_predictor``, ``saguaro_sampler``, ``device``, ``dtype``,
``vocab_size``, ``K``, ``fan_out``, ``_use_parallel_fanout``,
``_mtp_token_id``, ``_round_base_lens``, ``_swap_states``,
``_inflight_cache_build``, ``_last_*`` round state, the
``DraftServerSeqIdMixin`` helpers, and ``metrics``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.spec_decode.draft_data_models import (
    VerificationOutcome,
    decode,
)

logger = init_logger(__name__)


class DraftServerCacheBuildMixin:
    """Mixin: cache-build orchestration for ``DraftServer``."""

    # Sized so the merged tree decode comfortably fits within a single
    # forward at multi-VS load (B_total × entries_per_seq <= 504).
    MAX_BRANCHES = 504

    @staticmethod
    def _shrink_fan_out_to_budget(
        fan_out_list: list[int], B: int, max_branches: int,
    ) -> list[int]:
        """Proportionally scale ``fan_out_list`` so B × sum(...) <= budget.

        Preserves the geometric shape (earlier acceptance positions
        keep more candidates than later ones) and never lets any
        position drop below 1 unless we have to.
        """
        entries_per_seq = sum(fan_out_list)
        if B * entries_per_seq <= max_branches:
            return fan_out_list
        scale = max_branches / (B * entries_per_seq)
        shrunk = [max(1, int(f * scale)) for f in fan_out_list]
        while B * sum(shrunk) > max_branches:
            max_idx = max(range(len(shrunk)), key=lambda i: shrunk[i])
            if shrunk[max_idx] <= 1:
                break
            shrunk[max_idx] -= 1
        return shrunk

    async def _run_cache_build(
        self,
        batch_size: int,
        seq_ids: torch.Tensor,
        vs_id: str,
    ) -> None:
        """Background wrapper around ``_build_next_cache``.

        Invoked via ``asyncio.create_task`` after the SPECULATE response
        is sent so the serve loop can return to ``recv_multipart`` while
        GPU cache-building kernels run on the default stream. ZMQ recv
        and command decode for the next message overlap with this GPU
        work. Any subsequent handler awaits this task before mutating
        runner/cache state.

        ``vs_id`` scopes cache-reset and dedicated-block recycling so
        peer VSes' preserved cache entries survive this build.
        """
        runner = self.draft_model_runner
        if runner is None:
            return
        with torch.profiler.record_function(
            f"cache_build_B{batch_size}"
        ):
            # Snapshot _seq_lens around the build: tree decode mutates them
            # for its branch KV layout, and we need the per-seq lens to stay
            # at the end-of-round value for the next SPECULATE.
            saved = dict(runner._seq_lens)
            self._build_next_cache(batch_size, seq_ids, vs_id)
            runner._seq_lens = saved

    @staticmethod
    async def _await_cache_build_tasks(
        tasks: list[asyncio.Task],
    ) -> None:
        """Await all per-VS cache build tasks; log exceptions."""
        for t in tasks:
            try:
                await t
            except Exception:
                logger.exception(
                    "DraftServer per-VS cache build task failed."
                )

    async def _run_cache_build_merged(
        self,
        slice_metas: list[dict[str, Any]],
    ) -> None:
        """One merged cache build for the merged-SPECULATE path.

        Replaces what was N separate per-VS ``_run_cache_build_slice``
        tasks with a single forward over the concatenated batch.
        Per-VS scoping (``cache.reset_vs``, ``recycle_dedicated_blocks``,
        ``cache.populate``) is dispatched per slice; the heavy GPU work
        (glue_decode, allocate-and-copy-KV, tree_decode) runs once.

        Args:
            slice_metas: Per-VS dicts with keys vs_id, B, seq_ids,
                bonus_tokens, draft_tokens, draft_logits. Each tensor's
                first dim equals B.
        """
        runner = self.draft_model_runner
        if runner is None or not runner._model_loaded:
            return
        if len(slice_metas) == 0:
            return

        K = self.K
        if (
            self.cache is None
            or self.outcome_predictor is None
        ):
            return

        with torch.profiler.record_function(
            f"cache_build_merged_n{len(slice_metas)}"
        ):
            saved = dict(runner._seq_lens)
            try:
                # Reset and recycle per-VS upfront.
                for sm in slice_metas:
                    self.cache.reset_vs(sm["vs_id"])
                    runner.recycle_dedicated_blocks(sm["vs_id"])

                # Concatenate.
                seq_ids_cat = torch.cat(
                    [sm["seq_ids"] for sm in slice_metas], dim=0,
                )
                draft_tokens_cat = torch.cat(
                    [sm["draft_tokens"] for sm in slice_metas], dim=0,
                )
                draft_logits_cat = torch.cat(
                    [sm["draft_logits"] for sm in slice_metas], dim=0,
                )
                bonus_cat = torch.cat(
                    [sm["bonus_tokens"] for sm in slice_metas], dim=0,
                )
                B_total = seq_ids_cat.shape[0]
                seq_ids_list = seq_ids_cat.tolist()

                # Geometric fan-out (shared across VSes).
                fan_out_list = self._shrink_fan_out_to_budget(
                    list(self.outcome_predictor.fan_out_list),
                    B_total, self.MAX_BRANCHES,
                )
                entries_per_seq = sum(fan_out_list)
                N = B_total * entries_per_seq
                if N > self.MAX_BRANCHES or N == 0:
                    return
                max_fan_out = (
                    max(fan_out_list) if fan_out_list else 0
                )

                # Per-row glue input: bonus token for miss rows
                # (which received zero drafts), last drafted token
                # for hit rows (whose drafts came from the cache).
                merged_miss_mask = None
                if all("miss_mask" in sm for sm in slice_metas):
                    merged_miss_mask = torch.cat(
                        [sm["miss_mask"] for sm in slice_metas], dim=0,
                    )
                if (
                    merged_miss_mask is not None
                    and bool(merged_miss_mask.any().item())
                ):
                    glue_input = draft_tokens_cat[:, -1].clone()
                    glue_input[merged_miss_mask] = bonus_cat[
                        merged_miss_mask
                    ]
                else:
                    glue_input = draft_tokens_cat[:, -1]
                # ONE merged glue_decode (advances _seq_lens by 1 per seq).
                glue_logits = runner.glue_decode(
                    tokens=glue_input, seq_ids=seq_ids_cat,
                )

                # Zero-fallback miss rows: replace k=0 logits with
                # glue_logits so bonus candidates are real (see
                # _build_standalone_cache for full reasoning).
                if (
                    merged_miss_mask is not None
                    and bool(merged_miss_mask.any().item())
                ):
                    draft_logits_cat = draft_logits_cat.clone()
                    draft_logits_cat[merged_miss_mask, 0] = (
                        glue_logits[merged_miss_mask]
                    )

                post_glue_lens = {
                    sid: runner._seq_lens.get(sid, 0)
                    for sid in seq_ids_list
                }

                # ONE merged bonus selection.
                (
                    entry_batch_ids,
                    k_positions,
                    bonus_candidates,
                    _branches,
                ) = self._select_bonus_candidates(
                    B=B_total,
                    fan_out_list=fan_out_list,
                    max_fan_out=max_fan_out,
                    draft_logits=draft_logits_cat,
                    draft_tokens=draft_tokens_cat,
                    rec_tokens=bonus_cat,
                    glue_logits=glue_logits,
                )

                # Map each entry back to the originating VS via
                # entry_batch_ids[i] (which indexes the merged batch).
                # Build slice_owner[i] = vs_idx for each of the N
                # entries.
                vs_of_seq: list[int] = []
                for vs_idx, sm in enumerate(slice_metas):
                    vs_of_seq.extend([vs_idx] * sm["B"])
                # entry_batch_ids: [N] int64 mapping branch i -> batch
                # row in the concatenated batch.
                entry_batch_ids_cpu = entry_batch_ids.tolist()
                entry_owner = [
                    vs_of_seq[bi] for bi in entry_batch_ids_cpu
                ]

                # ONE merged block allocation. Block reservation is
                # per-VS, so split allocated blocks by entry_owner.
                bs = runner.block_size
                blocks_per_branch = (K + bs) // bs + 1
                total_needed = N * blocks_per_branch
                available = (
                    (runner.num_kv_blocks - runner._next_free_block)
                    + len(runner._free_list)
                )
                if available < total_needed:
                    # Pool exhausted; restore lens and abort.
                    for sid in seq_ids_list:
                        if sid in post_glue_lens:
                            runner._seq_lens[sid] = (
                                post_glue_lens[sid] - 1
                            )
                    return
                dedicated_blocks = [
                    runner._alloc_one_block()
                    for _ in range(total_needed)
                ]
                # Reserve dedicated blocks per VS (chunk
                # dedicated_blocks by entry owner).
                blocks_by_vs: dict[int, list[int]] = {
                    i: [] for i in range(len(slice_metas))
                }
                for n in range(N):
                    base = n * blocks_per_branch
                    blocks_by_vs[entry_owner[n]].extend(
                        dedicated_blocks[base:base + blocks_per_branch]
                    )
                for vs_idx, blks in blocks_by_vs.items():
                    if blks:
                        runner.reserve_dedicated_blocks(
                            blks, slice_metas[vs_idx]["vs_id"],
                        )

                # Build branch_block_tables and prefix_lens (same math
                # as _allocate_branch_blocks_and_copy_kv but using
                # already-allocated dedicated_blocks).
                M = runner.max_num_blocks
                base_lens_t = torch.tensor(
                    [
                        self._round_base_lens.get(
                            int(seq_ids_cat[b].item()), 0,
                        )
                        for b in range(B_total)
                    ],
                    dtype=torch.int64,
                    device=self.device,
                )
                prefix_lens = (
                    base_lens_t[entry_batch_ids] + 1 + k_positions
                )
                seq_ids_for_branches = (
                    seq_ids_cat[entry_batch_ids].to(torch.int64)
                )
                branch_block_tables = runner._block_table_gpu[
                    seq_ids_for_branches
                ].contiguous()
                first_write_blk = prefix_lens // bs
                ded_tensor = torch.tensor(
                    dedicated_blocks,
                    dtype=torch.int64,
                    device=self.device,
                ).view(N, blocks_per_branch)
                j_range = torch.arange(
                    blocks_per_branch,
                    device=self.device,
                    dtype=torch.int64,
                )
                tbl_indices = (
                    first_write_blk.unsqueeze(1) + j_range.unsqueeze(0)
                )
                valid = tbl_indices < M
                n_idx = (
                    torch.arange(N, device=self.device)
                    .unsqueeze(1)
                    .expand_as(tbl_indices)
                )
                branch_block_tables[
                    n_idx[valid], tbl_indices[valid].to(torch.int64),
                ] = ded_tensor[valid].to(torch.int32)

                # KV copy from parent into newly-reserved blocks.
                parent_tables = runner._block_table_gpu[
                    seq_ids_for_branches
                ]
                src_indices = tbl_indices.clamp(max=M - 1)
                src_block_ids = parent_tables[
                    n_idx, src_indices.to(torch.int64),
                ].to(torch.int64)
                dst_block_ids = ded_tensor
                copy_mask = (
                    valid & (src_block_ids != dst_block_ids)
                )
                if copy_mask.any() and runner.kv_caches is not None:
                    src_flat = src_block_ids[copy_mask]
                    dst_flat = dst_block_ids[copy_mask]
                    for layer_kv in runner.kv_caches:
                        # Layout (num_blocks, 2, block_size, num_kv_heads,
                        # head_dim); block dim is 0.
                        layer_kv[dst_flat] = layer_kv[src_flat]

                # ONE merged tree decode (or parallel fanout).
                if self._use_parallel_fanout:
                    all_tokens, all_logits = self._run_parallel_fanout(
                        runner=runner,
                        N=N,
                        K=K,
                        seq_ids=seq_ids_cat,
                        entry_batch_ids=entry_batch_ids,
                        prefix_lens=prefix_lens,
                        branch_block_tables=branch_block_tables,
                        bonus_candidates=bonus_candidates,
                    )
                    cleanup_logits = self._parallel_fanout_kv_cleanup(
                        runner=runner,
                        N=N,
                        K=K,
                        seq_ids=seq_ids_cat,
                        entry_batch_ids=entry_batch_ids,
                        prefix_lens=prefix_lens,
                        branch_block_tables=branch_block_tables,
                        all_tokens=all_tokens,
                    )
                    if cleanup_logits is not None:
                        # Replace contaminated mask-derived logits at
                        # depths 1+ with cleanup's real-token logits.
                        # Depth 0 (bonus NTP) was never contaminated.
                        all_logits = all_logits.clone()
                        all_logits[:, 1:K, :] = cleanup_logits
                else:
                    all_tokens, all_logits = self._run_tree_decode(
                        runner=runner,
                        N=N,
                        K=K,
                        seq_ids=seq_ids_cat,
                        entry_batch_ids=entry_batch_ids,
                        prefix_lens=prefix_lens,
                        branch_block_tables=branch_block_tables,
                        bonus_candidates=bonus_candidates,
                    )

                # Split populate per VS by entry_owner.
                entry_owner_t = torch.tensor(
                    entry_owner, dtype=torch.int64, device=self.device,
                )
                seq_ids_per_branch = seq_ids_cat[entry_batch_ids]
                for vs_idx, sm in enumerate(slice_metas):
                    mask = entry_owner_t == vs_idx
                    if not mask.any():
                        continue
                    self.cache.populate(
                        seq_ids=seq_ids_per_branch[mask],
                        k_positions=k_positions[mask],
                        bonus_tokens=bonus_candidates[mask],
                        draft_tokens=all_tokens[mask],
                        draft_logits=all_logits[mask],
                        branch_block_tables=branch_block_tables[mask],
                        prefix_lens=prefix_lens[mask],
                        vs_id=sm["vs_id"],
                    )

                # Undo glue's +1 on _seq_lens so next round's
                # reconciliation starts from the same base.
                for sid in seq_ids_list:
                    if sid in post_glue_lens:
                        runner._seq_lens[sid] = post_glue_lens[sid] - 1

            finally:
                # Restore caller's _seq_lens snapshot (the per-round
                # end-of-round value), not glue's mutated state.
                runner._seq_lens = saved

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _build_next_cache(
        self,
        batch_size: int,
        seq_ids: torch.Tensor,
        vs_id: str,
    ) -> None:
        """Pre-compute the speculation cache for the NEXT round.

        Scoped to a single verify server: only ``vs_id``'s partition
        of the SpeculationCache is reset and only ``vs_id``'s dedicated
        blocks are recycled. Peer VSes' preserved entries stay intact.
        """
        if self.cache is not None:
            self.cache.reset_vs(vs_id)

        runner = self.draft_model_runner
        if runner is None or not runner._model_loaded:
            return
        if (
            self._last_draft_tokens is None
            or self._last_draft_logits is None
        ):
            return

        B = batch_size
        K = self.K

        # Geometric fan-out: per-position candidate counts computed by
        # the OutcomePredictor (earlier acceptance positions get more
        # budget since they're more likely to be the actual outcome).
        fan_out_list = self._shrink_fan_out_to_budget(
            list(self.outcome_predictor.fan_out_list),
            B, self.MAX_BRANCHES,
        )
        entries_per_seq = sum(fan_out_list)
        N = B * entries_per_seq
        if N > self.MAX_BRANCHES:
            return

        max_fan_out = max(fan_out_list) if fan_out_list else 0
        seq_ids_list = seq_ids.tolist()

        self._build_standalone_cache(
            B, K, fan_out_list, max_fan_out, N,
            seq_ids, seq_ids_list, runner,
            self._last_draft_tokens,
            self._last_draft_logits,
            self._last_bonus_tokens,
            vs_id,
            miss_mask=self._last_miss_mask,
        )

    def _select_bonus_candidates(
        self,
        B: int,
        fan_out_list: list[int],
        max_fan_out: int,
        draft_logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        rec_tokens: torch.Tensor,
        glue_logits: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, int,
    ]:
        """Pick top-F bonus-token candidates per acceptance position.

        Masks out the tokens we already drafted so cache entries cover
        distinct bonus branches, then returns the flattened
        (entry_batch_ids, k_positions, bonus_candidates) triple plus
        branches_per_seq.
        """
        outcome_logits = torch.cat(
            [draft_logits, glue_logits.unsqueeze(1)], dim=1
        )
        outcome_tokens = torch.cat(
            [rec_tokens.unsqueeze(1), draft_tokens], dim=1
        )

        masked_logits = outcome_logits.clone()
        masked_logits[:, :-1, :] = masked_logits[:, :-1, :].scatter(
            dim=2,
            index=outcome_tokens[:, 1:].unsqueeze(2),
            value=float("-inf"),
        )
        _, topk_indices = torch.topk(masked_logits, max_fan_out, dim=-1)

        branches_per_seq = sum(fan_out_list)
        per_seq_k: list[torch.Tensor] = []
        per_seq_cand_slots: list[torch.Tensor] = []
        for k, F_k in enumerate(fan_out_list):
            if F_k <= 0:
                continue
            per_seq_k.append(torch.full(
                (F_k,), k, dtype=torch.int64, device=self.device
            ))
            per_seq_cand_slots.append(torch.arange(
                F_k, dtype=torch.int64, device=self.device,
            ))
        empty = torch.zeros(0, dtype=torch.int64, device=self.device)
        per_seq_k_flat = torch.cat(per_seq_k) if per_seq_k else empty
        per_seq_cand_flat = (
            torch.cat(per_seq_cand_slots) if per_seq_cand_slots else empty
        )

        k_positions = per_seq_k_flat.unsqueeze(0).expand(
            B, branches_per_seq
        ).reshape(-1)
        entry_batch_ids = torch.arange(
            B, device=self.device, dtype=torch.int64,
        ).unsqueeze(1).expand(B, branches_per_seq).reshape(-1)
        cand_slots_full = per_seq_cand_flat.unsqueeze(0).expand(
            B, branches_per_seq
        ).reshape(-1)
        bonus_candidates = topk_indices[
            entry_batch_ids, k_positions, cand_slots_full
        ]
        return entry_batch_ids, k_positions, bonus_candidates, branches_per_seq

    def _allocate_branch_blocks_and_copy_kv(
        self,
        runner: Any,
        vs_id: str,
        N: int,
        K: int,
        seq_ids: torch.Tensor,
        entry_batch_ids: torch.Tensor,
        k_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Reserve dedicated blocks for N branches and copy parent KV in.

        Returns (branch_block_tables, prefix_lens) on success, or None
        if the block pool is exhausted.
        """
        bs = runner.block_size
        M = runner.max_num_blocks
        blocks_per_branch = (K + bs) // bs + 1
        total_needed = N * blocks_per_branch
        available = (
            (runner.num_kv_blocks - runner._next_free_block)
            + len(runner._free_list)
        )
        if available < total_needed:
            return None

        dedicated_blocks = [
            runner._alloc_one_block() for _ in range(total_needed)
        ]
        runner.reserve_dedicated_blocks(dedicated_blocks, vs_id)

        B = seq_ids.shape[0]
        base_lens_t = torch.tensor(
            [
                self._round_base_lens.get(int(seq_ids[b].item()), 0)
                for b in range(B)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        prefix_lens = base_lens_t[entry_batch_ids] + 1 + k_positions

        seq_ids_for_branches = seq_ids[entry_batch_ids].to(torch.int64)
        branch_block_tables = runner._block_table_gpu[
            seq_ids_for_branches
        ].contiguous()

        first_write_blk = prefix_lens // bs
        ded_tensor = torch.tensor(
            dedicated_blocks, dtype=torch.int64, device=self.device
        ).view(N, blocks_per_branch)

        j_range = torch.arange(
            blocks_per_branch, device=self.device, dtype=torch.int64
        )
        tbl_indices = first_write_blk.unsqueeze(1) + j_range.unsqueeze(0)
        valid = tbl_indices < M
        n_idx = (
            torch.arange(N, device=self.device)
            .unsqueeze(1)
            .expand_as(tbl_indices)
        )
        branch_block_tables[
            n_idx[valid], tbl_indices[valid].to(torch.int64)
        ] = ded_tensor[valid].to(torch.int32)

        parent_tables = runner._block_table_gpu[seq_ids_for_branches]
        src_indices = tbl_indices.clamp(max=M - 1)
        src_block_ids = parent_tables[
            n_idx, src_indices.to(torch.int64)
        ].to(torch.int64)
        dst_block_ids = ded_tensor
        copy_mask = valid & (src_block_ids != dst_block_ids)
        if copy_mask.any() and runner.kv_caches is not None:
            src_flat = src_block_ids[copy_mask]
            dst_flat = dst_block_ids[copy_mask]
            for layer_kv in runner.kv_caches:
                # Layout (num_blocks, 2, block_size, num_kv_heads, head_dim);
                # block dim is 0.
                layer_kv[dst_flat] = layer_kv[src_flat]

        return branch_block_tables, prefix_lens

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

    def _build_standalone_cache(
        self,
        B: int, K: int,
        fan_out_list: list[int],
        max_fan_out: int,
        N: int,
        seq_ids: torch.Tensor,
        seq_ids_list: list[int],
        runner: Any,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor,
        rec_tokens: torch.Tensor,
        vs_id: str,
        miss_mask: torch.Tensor | None = None,
    ) -> None:
        """Build speculation cache for standalone draft models.

        Uses dedicated blocks with KV copy. Fan-out is per-position
        (geometric allocation), not uniform. Dedicated-block
        allocation is scoped to ``vs_id`` so peer VSes' preserved
        cache entries keep pointing at live KV data.

        ``miss_mask`` marks rows where speculate sent zero drafts.
        For those rows, glue_decode uses the bonus token (not
        draft_tokens[:, -1]) to compute the next-position prediction;
        ``_seq_lens`` for miss rows is at ``base`` so glue writes the
        bonus's KV at base.
        """
        runner.recycle_dedicated_blocks(vs_id)

        # Glue decode gives us the K+1th position's logits, and
        # advances _seq_lens by one — we undo that at the end.
        # Per-row glue input: draft_tokens[:,-1] for hit rows (came
        # from the cache), bonus token for miss rows (got zeros).
        if miss_mask is not None and bool(miss_mask.any().item()):
            glue_input = draft_tokens[:, -1].clone()
            glue_input[miss_mask] = rec_tokens[miss_mask]
        else:
            glue_input = draft_tokens[:, -1]
        glue_logits = runner.glue_decode(
            tokens=glue_input, seq_ids=seq_ids
        )

        # Zero-fallback miss rows: draft_logits is all zeros (no JIT
        # ran). _select_bonus_candidates would compute top-F over zeros
        # → garbage token-ids → unmatchable cache entries. Replace the
        # k=0 row with glue_logits (real prediction at base+1) so the
        # k=0 branch's bonus candidates align with what the verifier
        # will sample next round. Higher-k entries are unreachable
        # (verifier accepted 0 of zeros) so we leave them as zeros.
        if miss_mask is not None and bool(miss_mask.any().item()):
            draft_logits = draft_logits.clone()
            draft_logits[miss_mask, 0] = glue_logits[miss_mask]

        post_glue_lens = {
            sid: runner._seq_lens.get(sid, 0) for sid in seq_ids_list
        }

        entry_batch_ids, k_positions, bonus_candidates, _branches = (
            self._select_bonus_candidates(
                B=B,
                fan_out_list=fan_out_list,
                max_fan_out=max_fan_out,
                draft_logits=draft_logits,
                draft_tokens=draft_tokens,
                rec_tokens=rec_tokens,
                glue_logits=glue_logits,
            )
        )

        alloc = self._allocate_branch_blocks_and_copy_kv(
            runner=runner,
            vs_id=vs_id,
            N=N,
            K=K,
            seq_ids=seq_ids,
            entry_batch_ids=entry_batch_ids,
            k_positions=k_positions,
        )
        if alloc is None:
            # Block pool exhausted; skip cache build and restore seq lens.
            for sid in seq_ids_list:
                if sid in post_glue_lens:
                    runner._seq_lens[sid] = post_glue_lens[sid] - 1
            return
        branch_block_tables, prefix_lens = alloc

        if self._use_parallel_fanout:
            all_tokens, all_logits = self._run_parallel_fanout(
                runner=runner,
                N=N,
                K=K,
                seq_ids=seq_ids,
                entry_batch_ids=entry_batch_ids,
                prefix_lens=prefix_lens,
                branch_block_tables=branch_block_tables,
                bonus_candidates=bonus_candidates,
            )
            cleanup_logits = self._parallel_fanout_kv_cleanup(
                runner=runner,
                N=N,
                K=K,
                seq_ids=seq_ids,
                entry_batch_ids=entry_batch_ids,
                prefix_lens=prefix_lens,
                branch_block_tables=branch_block_tables,
                all_tokens=all_tokens,
            )
            if cleanup_logits is not None:
                all_logits = all_logits.clone()
                all_logits[:, 1:K, :] = cleanup_logits
        else:
            all_tokens, all_logits = self._run_tree_decode(
                runner=runner,
                N=N,
                K=K,
                seq_ids=seq_ids,
                entry_batch_ids=entry_batch_ids,
                prefix_lens=prefix_lens,
                branch_block_tables=branch_block_tables,
                bonus_candidates=bonus_candidates,
            )

        self.cache.populate(
            seq_ids=seq_ids[entry_batch_ids],
            k_positions=k_positions,
            bonus_tokens=bonus_candidates,
            draft_tokens=all_tokens,
            draft_logits=all_logits,
            branch_block_tables=branch_block_tables,
            prefix_lens=prefix_lens,
            vs_id=vs_id,
        )

        # Undo glue's +1 on _seq_lens so next round's reconciliation
        # starts from the same base as before this cache build.
        for sid in seq_ids_list:
            if sid in post_glue_lens:
                runner._seq_lens[sid] = post_glue_lens[sid] - 1

    # ------------------------------------------------------------------
    # Prefill and free_seq command handlers
    # ------------------------------------------------------------------

