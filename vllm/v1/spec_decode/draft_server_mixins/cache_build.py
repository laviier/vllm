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
            # Apply any swap deferred from the synchronous SPECULATE
            # path. Mutates _seq_lens / block tables and recycles
            # displaced blocks; running it here (rather than on the
            # SPECULATE critical path) overlaps with the verifier's
            # target forward.
            pending = self._pending_swap
            self._pending_swap = None
            if pending is not None:
                # Annotation label drops the hit count (would force a
                # CPU↔GPU sync just to format the string); inspect the
                # tensor in the trace if you need the per-round count.
                with torch.profiler.record_function("deferred_swap"):
                    self._apply_pending_swap(runner=runner, **pending)
            # Fused cleanup+glue runs whether or not there are hits — on
            # cold rounds with all-miss it still produces glue logits for
            # _select_bonus_candidates (replaces today's runner.glue_decode
            # call inside _build_next_cache).
            self._fused_cleanup_and_glue(runner, pending)

            # Snapshot _seq_lens around the build: tree decode mutates them
            # for its branch KV layout, and we need the per-seq lens to stay
            # at the end-of-round value for the next SPECULATE.
            saved = dict(runner._seq_lens)
            self._build_next_cache(batch_size, seq_ids, vs_id)
            runner._seq_lens = saved

    def _fused_cleanup_and_glue(
        self,
        runner: Any,
        pending: dict[str, Any] | None,
    ) -> None:
        """Run cleanup (KV refresh on hit seqs) and glue_decode in one
        fused varlen forward, before ``_build_next_cache`` runs.

        Hit seqs contribute K tokens each: K-1 cleanup tokens at slots
        ``prefix+1..prefix+K-1`` (overwriting parallel_fanout's mask
        KVs), plus 1 glue token at slot ``prefix+K = _seq_lens[sid]``
        (the bonus_t's home in seq main).

        Miss seqs contribute 1 glue token each at slot ``_seq_lens[sid]``
        (zero-fallback recovery; input is the bonus token).

        Stashes the per-seq glue logits as ``self._pending_glue_logits``
        for the upcoming ``_select_bonus_candidates``, and splices the
        per-hit cleanup logits into ``self._last_draft_logits[hit, 1:K]``.

        The fused forward also advances ``_seq_lens[sid] += 1`` for every
        seq, mirroring the side effect today's separate ``glue_decode``
        has on the runner.
        """
        if (
            self._last_draft_tokens is None
            or self._last_draft_logits is None
            or self._last_bonus_tokens is None
        ):
            # Cold start before the first SPECULATE round populates
            # _last_*. Caller falls back to runner.glue_decode for the
            # cold path; nothing to fuse here.
            return
        # _last_miss_mask aligns with this round's seqs (stashed by the
        # SPECULATE handler at the same time as _last_*). The zeros
        # fallback is unreachable under the early return above (any
        # round with miss rows also stashed _last_miss_mask), but kept
        # as defense in depth so a future change that decouples these
        # stashes doesn't silently send mask=False through to glue
        # input selection.
        miss_mask_full = (
            self._last_miss_mask if self._last_miss_mask is not None
            else torch.zeros_like(self._last_bonus_tokens, dtype=torch.bool)
        )
        bonus_tokens = self._last_bonus_tokens                  # [B]
        last_draft_last_col = self._last_draft_tokens[:, -1]    # [B]
        B = bonus_tokens.shape[0]
        K = self.K
        D = K - 1

        # Per-seq glue position = _seq_lens[sid]. Read in seq_ids order.
        if pending is not None:
            seq_ids_full = pending["seq_ids"]                   # [B]
        else:
            # No swap pending (cold start / all-miss round): the SPECULATE
            # handler stashed _last_spec_seq_ids at the same time as the
            # other _last_* fields.
            seq_ids_full = self._last_spec_seq_ids
            if seq_ids_full is None:
                return
        if seq_ids_full.shape[0] != B:
            return

        seq_ids_list = seq_ids_full.tolist()
        glue_positions_cpu = [
            int(runner._seq_lens.get(int(sid), 0)) for sid in seq_ids_list
        ]
        glue_positions = torch.tensor(
            glue_positions_cpu, dtype=torch.int64, device=self.device,
        )
        # Per-seq glue input: bonus for miss rows (zero-fallback), last
        # cached draft for hit rows.
        glue_input = torch.where(
            miss_mask_full, bonus_tokens, last_draft_last_col,
        ).to(torch.int32)

        # Hit-side cleanup payload (only meaningful when parallel fanout
        # is active and last round's branches actually got installed).
        do_cleanup = (
            self._use_parallel_fanout
            and K > 1
            and pending is not None
        )
        if do_cleanup:
            cache_hits = pending["cache_hits"]
            hit_prefix_lens = pending["hit_prefix_lens"]            # [H]
            hit_mask = cache_hits.bool()
            H = int(cache_hits.sum().item())
        else:
            hit_mask = torch.zeros(B, dtype=torch.bool, device=self.device)
            hit_prefix_lens = None
            H = 0

        # ---- Build fused varlen input ----
        # Layout per seq, in seq_ids_full order:
        #   hit seq i (if do_cleanup):
        #     positions[seq_i] = [prefix_i+1, prefix_i+2, ..., prefix_i+K-1, glue_pos_i]
        #     inputs   [seq_i] = [_last_draft_tokens[i,0..K-2], glue_input[i]]
        #     count = K
        #   miss/non-hit seq i:
        #     positions[seq_i] = [glue_pos_i]
        #     inputs   [seq_i] = [glue_input[i]]
        #     count = 1
        # We build flat tensors of total length (H * K + (B - H) * 1).
        if H > 0:
            hit_indices = hit_mask.nonzero(as_tuple=True)[0]   # [H]
            assert hit_prefix_lens is not None  # for type checker
            depth_offsets = torch.arange(
                1, K, device=self.device, dtype=torch.int64,
            )                                                   # [K-1]
            cleanup_positions = (
                hit_prefix_lens.to(torch.int64).unsqueeze(1)
                + depth_offsets.unsqueeze(0)
            )                                                   # [H, K-1]
            cleanup_inputs = self._last_draft_tokens[hit_indices, :D]
            cleanup_inputs = cleanup_inputs.to(torch.int32)     # [H, K-1]
            hit_glue_pos = glue_positions[hit_indices].unsqueeze(1)  # [H,1]
            hit_glue_inp = glue_input[hit_indices].unsqueeze(1)      # [H,1]
            # Per-hit-seq segment: [cleanup..., glue]
            hit_positions = torch.cat(
                [cleanup_positions, hit_glue_pos], dim=1,
            ).reshape(-1)                                       # [H*K]
            hit_inputs = torch.cat(
                [cleanup_inputs, hit_glue_inp], dim=1,
            ).reshape(-1)                                       # [H*K]
        else:
            hit_indices = torch.empty(0, dtype=torch.int64, device=self.device)
            hit_positions = torch.empty(0, dtype=torch.int64, device=self.device)
            hit_inputs = torch.empty(0, dtype=torch.int32, device=self.device)

        # Non-hit (miss + first-round-no-swap) seqs: 1 glue token each.
        nonhit_mask = ~hit_mask
        nonhit_indices = nonhit_mask.nonzero(as_tuple=True)[0]
        nonhit_positions = glue_positions[nonhit_indices]
        nonhit_inputs = glue_input[nonhit_indices]
        nB = int(nonhit_indices.shape[0])

        # Concatenate: hits first, then non-hits.
        positions = torch.cat([hit_positions, nonhit_positions], dim=0)
        input_ids = torch.cat([hit_inputs, nonhit_inputs], dim=0)
        N_fused = positions.shape[0]
        if N_fused == 0:
            return

        # seq_lens (causal): position + 1.
        seq_lens = (positions + 1).to(torch.int32)

        # Per-token block_tables: each token attends over its seq's main
        # block table (post-swap for hits, unchanged for non-hits).
        per_token_seq_ids = torch.cat([
            seq_ids_full[hit_indices].repeat_interleave(K),
            seq_ids_full[nonhit_indices],
        ], dim=0)
        block_tables = runner._block_table_gpu[per_token_seq_ids]

        max_context_hint = int(positions.max().item()) + 1

        with torch.profiler.record_function(
            f"fused_cleanup_glue_H{H}_M{nB}"
        ):
            logits_flat = runner.tree_decode_step(
                input_ids=input_ids,
                positions=positions,
                seq_lens=seq_lens,
                seq_ids_expanded=per_token_seq_ids,
                block_tables=block_tables,
                max_seq_len_hint=max_context_hint,
            )

        # ---- Split outputs ----
        # First H*K rows: per-hit segments [cleanup_0..K-2, glue].
        # Next nB rows: per-non-hit glue.
        glue_logits = torch.zeros(
            B, self.vocab_size, dtype=self.dtype, device=self.device,
        )
        if H > 0:
            hit_logits = logits_flat[: H * K].view(H, K, -1)
            cleanup_logits = hit_logits[:, :D, :]               # [H, K-1, V]
            hit_glue_logits = hit_logits[:, D, :]               # [H, V]
            self._last_draft_logits = self._last_draft_logits.clone()
            self._last_draft_logits[hit_indices, 1:K, :] = cleanup_logits
            glue_logits[hit_indices] = hit_glue_logits
        if nB > 0:
            nonhit_glue_logits = logits_flat[H * K :]            # [nB, V]
            glue_logits[nonhit_indices] = nonhit_glue_logits

        self._pending_glue_logits = glue_logits

        # Mirror glue_decode's _seq_lens advance (one per seq).
        for sid, pos in zip(seq_ids_list, glue_positions_cpu):
            runner._seq_lens[int(sid)] = pos + 1

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
            # Merged path runs swap_block_tables synchronously in the
            # SPECULATE handler (deferring it regressed multi-VS TPOT
            # by 1-3 %). The fused cleanup+glue forward then refreshes
            # mask KVs and produces glue logits in one varlen forward.
            pending_merged = self._pending_swap_merged
            self._pending_swap_merged = None
            self._fused_cleanup_and_glue(runner, pending_merged)

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
                # Note: sm["draft_logits"] is intentionally NOT
                # concatenated here. ``_select_bonus_candidates`` below
                # reads ``self._last_draft_logits`` instead, which
                # carries the cleanup splice from
                # ``_fused_cleanup_and_glue`` (positions 1..K-1 of HIT
                # rows refreshed with real-context logits). Feeding
                # the un-spliced per-VS slice would produce nonsense
                # bonus candidates at HIT rows because mask-context
                # logits at positions 1..K-1 are near-flat for the
                # parallel-MTP drafter.
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
                if self._pending_glue_logits is not None:
                    glue_logits = self._pending_glue_logits
                    self._pending_glue_logits = None
                else:
                    # Unconditional ``torch.where`` avoids the
                    # ``mask.any().item()`` host sync; result is just
                    # the last-col tokens when the mask is all-False.
                    # When all 3V cb_merged_n3 rounds drained on this
                    # sync, it cost ~0.5-1 ms each.
                    if merged_miss_mask is not None:
                        glue_input = torch.where(
                            merged_miss_mask,
                            bonus_cat,
                            draft_tokens_cat[:, -1],
                        )
                    else:
                        glue_input = draft_tokens_cat[:, -1]
                    # Advances _seq_lens by 1 per seq (mirrored by the
                    # fused-prologue path).
                    glue_logits = runner.glue_decode(
                        tokens=glue_input, seq_ids=seq_ids_cat,
                    )

                # Use ``self._last_draft_logits`` (cleanup-spliced) as
                # the source for ``_select_bonus_candidates``; the per-
                # VS slice in slice_metas is the un-spliced version
                # from the speculate handler (see note where it would
                # have been concatenated above).
                draft_logits_for_select = self._last_draft_logits
                # Zero-fallback miss rows: replace k=0 logits with
                # glue_logits so bonus candidates are real (see
                # _build_standalone_cache for full reasoning). Run
                # unconditionally — masked assignment is a no-op when
                # the mask is all-False; gating with .any().item() was
                # a CPU↔GPU sync per round.
                if merged_miss_mask is not None:
                    draft_logits_for_select = (
                        draft_logits_for_select.clone()
                    )
                    draft_logits_for_select[merged_miss_mask, 0] = (
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
                    draft_logits=draft_logits_for_select,
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
                # entry_batch_ids is a deterministic
                # ``arange(B_total).repeat_interleave(entries_per_seq)``
                # pattern produced by ``_select_bonus_candidates``; we
                # reconstruct the host-side equivalent without a GPU→CPU
                # sync. The prior ``entry_batch_ids.tolist()`` showed
                # up as ~1.17 ms in the inter-phase band of every
                # merged cb cycle because it forced the
                # ``_select_bonus_candidates`` GPU queue to drain.
                entry_owner = [
                    vs_of_seq[b] for b in range(B_total)
                    for _ in range(entries_per_seq)
                ]
                # Hoisted: build the GPU tensor now so its CPU→GPU
                # copy queues alongside the upcoming block alloc /
                # KV-copy / parallel_fanout work instead of after them.
                # Originally created at line ~601 right before the
                # populate loop, where the trace showed it stalling
                # 17 ms on cudaStreamSynchronize waiting for
                # parallel_fanout's tail kernels to drain.
                entry_owner_t = torch.tensor(
                    entry_owner, dtype=torch.int64, device=self.device,
                )

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
                dedicated_blocks = runner._alloc_n_blocks(total_needed)
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
                # seq_ids_list was already materialized above (line ~376).
                # Use it directly instead of per-element seq_ids_cat[b].item(),
                # which adds B_total CPU↔GPU syncs.
                base_lens_t = torch.tensor(
                    [
                        self._round_base_lens.get(sid, 0)
                        for sid in seq_ids_list
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
                # Read parent block IDs BEFORE mutating branch_block_tables:
                # at this point it's still a clean copy of the parent table,
                # so we can skip a second _block_table_gpu gather.
                src_indices_i64 = tbl_indices.clamp(max=M - 1).to(torch.int64)
                src_block_ids = branch_block_tables[
                    n_idx, src_indices_i64,
                ].to(torch.int64)

                branch_block_tables[
                    n_idx[valid], tbl_indices[valid].to(torch.int64),
                ] = ded_tensor[valid].to(torch.int32)

                # KV copy from parent into newly-reserved blocks.
                # Run unconditionally — when copy_mask is all-False
                # the per-layer index_put_ is a 0-element no-op, and
                # in practice copy_mask is almost always True (parent
                # and dedicated block IDs differ for the blocks being
                # written). The prior ``copy_mask.any()`` guard cost a
                # full GPU-queue sync just to skip 32 empty kernels.
                dst_block_ids = ded_tensor
                copy_mask = (
                    valid & (src_block_ids != dst_block_ids)
                )
                if runner.kv_caches is not None:
                    src_flat = src_block_ids[copy_mask]
                    dst_flat = dst_block_ids[copy_mask]
                    for layer_kv in runner.kv_caches:
                        # Layout (num_blocks, 2, block_size, num_kv_heads,
                        # head_dim); block dim is 0.
                        layer_kv[dst_flat] = layer_kv[src_flat]

                # ONE merged tree decode (or parallel fanout). Mask KVs
                # at depths 1..K-1 are dirty in the dedicated blocks
                # we just wrote; they get cleaned next round in
                # ``_fused_cleanup_and_glue`` for whichever branch
                # wins the next-round lookup.
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

                # Split populate per VS by entry_owner. ``entry_owner_t``
                # was hoisted above to overlap its CPU→GPU copy with
                # earlier GPU work.
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
        # Fast path: ``fan_out_list`` matches the OutcomePredictor's
        # configured value (the common case — shrinkage only fires
        # when ``B * sum(fan_out) > MAX_BRANCHES``, which we never
        # reach at the supported concurrencies). Reuse the flat
        # indexing tensors precomputed at init instead of rebuilding
        # them every round.
        if (
            self.outcome_predictor is not None
            and fan_out_list == self.outcome_predictor.fan_out_list
        ):
            per_seq_k_flat = self.outcome_predictor.per_seq_k_flat
            per_seq_cand_flat = self.outcome_predictor.per_seq_cand_flat
        else:
            per_seq_k_chunks: list[torch.Tensor] = []
            per_seq_cand_chunks: list[torch.Tensor] = []
            for k, F_k in enumerate(fan_out_list):
                if F_k <= 0:
                    continue
                per_seq_k_chunks.append(torch.full(
                    (F_k,), k, dtype=torch.int64, device=self.device
                ))
                per_seq_cand_chunks.append(torch.arange(
                    F_k, dtype=torch.int64, device=self.device,
                ))
            empty = torch.zeros(0, dtype=torch.int64, device=self.device)
            per_seq_k_flat = (
                torch.cat(per_seq_k_chunks) if per_seq_k_chunks else empty
            )
            per_seq_cand_flat = (
                torch.cat(per_seq_cand_chunks) if per_seq_cand_chunks
                else empty
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

        dedicated_blocks = runner._alloc_n_blocks(total_needed)
        runner.reserve_dedicated_blocks(dedicated_blocks, vs_id)

        B = seq_ids.shape[0]
        # Materialize seq_ids once instead of per-element .item() calls
        # (each would be a CPU↔GPU sync).
        seq_ids_list = seq_ids.tolist()
        base_lens_t = torch.tensor(
            [
                self._round_base_lens.get(sid, 0)
                for sid in seq_ids_list
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
        # Read parent block IDs BEFORE mutating branch_block_tables:
        # at this point it's still a clean copy of the parent table,
        # so we can skip a second _block_table_gpu gather.
        src_indices_i64 = tbl_indices.clamp(max=M - 1).to(torch.int64)
        src_block_ids = branch_block_tables[
            n_idx, src_indices_i64,
        ].to(torch.int64)

        branch_block_tables[
            n_idx[valid], tbl_indices[valid].to(torch.int64)
        ] = ded_tensor[valid].to(torch.int32)

        dst_block_ids = ded_tensor
        copy_mask = valid & (src_block_ids != dst_block_ids)
        # Run unconditionally — see note on the merged-path variant.
        if runner.kv_caches is not None:
            src_flat = src_block_ids[copy_mask]
            dst_flat = dst_block_ids[copy_mask]
            for layer_kv in runner.kv_caches:
                # Layout (num_blocks, 2, block_size, num_kv_heads, head_dim);
                # block dim is 0.
                layer_kv[dst_flat] = layer_kv[src_flat]

        return branch_block_tables, prefix_lens

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
        # Glue logits and the +1 _seq_lens advance were already produced
        # by ``_fused_cleanup_and_glue`` in the cache_build prologue.
        # Fall back to a fresh glue_decode if the prologue didn't run
        # (e.g. cold start before any cache hits exist).
        if self._pending_glue_logits is not None:
            glue_logits = self._pending_glue_logits
            self._pending_glue_logits = None
        else:
            # Unconditional ``torch.where`` avoids the host sync;
            # result is the last-col tokens when the mask is all-False.
            # Mirrors the fix on the merged path.
            if miss_mask is not None:
                glue_input = torch.where(
                    miss_mask, rec_tokens, draft_tokens[:, -1]
                )
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
        # Run unconditionally — the masked assignment is a no-op when
        # miss_mask is all-False, and the prior gating .any().item()
        # was a CPU↔GPU sync per round.
        if miss_mask is not None:
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
            # Mask KVs at depths 1..K-1 are dirty in the dedicated blocks
            # we write here; they get cleaned next round in
            # ``_fused_cleanup_and_glue`` for the branch that wins
            # the next lookup.
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

