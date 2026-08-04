# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SPECULATE command handlers for ``DraftServer``.

This is the heart of the per-round draft loop. ``_handle_speculation``
runs the single-VS path (lookup → swap_hits or zero-fill on miss →
respond → schedule cache_build for next round), and the merged variants
(``_handle_speculation_merged*``) collapse N coincident SPECULATEs from
distinct VSes into one batched forward + one merged cache build.

Expects the consumer to expose: ``draft_model_runner``, ``cache``,
``device``, ``dtype``, ``vocab_size``, ``K``, ``metrics``,
``_round_base_lens``, ``_swap_states``, ``_last_*`` round state,
``_inflight_cache_build``, plus the
``DraftServerSeqIdMixin``/``DraftServerTransportMixin``/
``DraftServerCacheBuildMixin`` helpers.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.spec_decode.draft_data_models import (
    DraftCommand,
    VerificationOutcome,
    decode,
)

logger = init_logger(__name__)


class DraftServerSpeculateMixin:
    """Mixin: SPECULATE handlers (single-VS + cross-VS merged)."""

    async def _handle_speculation(
        self,
        verify_server_id: str,
        identity: bytes,
        outcome: VerificationOutcome,
    ) -> None:
        """Handle SPECULATE command with hybrid swap+JIT strategy.

        Replicates the ``DisaggDraftWorker._handle_speculation`` flow
        using the server's own components, decoupled from the NCCL
        command loop.  The steps are:

        1. Receive tensor payloads out-of-band (NCCL, matching the
           deterministic send order in ``ZmqDraftConnector``).
        2. Reconcile ``_seq_lens`` from this round's k_accepted.
        3. Cache lookup via ``SpeculationCache.lookup``.
        4. Hybrid swap+zero-fill: cache hits use cached tokens; misses
           get zero drafts (cache_build seeds entries for next round).
        5. Send ``SpeculationResponse`` metadata over ZMQ and tensor
           payloads over NCCL.
        6. Build speculation cache for the NEXT round (async overlap).

        On error, sends a fallback response (all zeros) so the verify
        server does not hang.
        """
        B = outcome.batch_size
        logger.debug(
            "DraftServer SPECULATE from %s, batch_size=%d",
            verify_server_id,
            B,
        )

        with torch.profiler.record_function(f"speculate_B{B}"):
            # Block until the previous round's cache build finishes.
            # Cache build mutates runner._seq_lens, block tables, the
            # SpeculationCache contents, and issues GPU kernels that share
            # the default stream with this handler's JIT/glue work — so
            # we must serialize them. Awaiting here (rather than in the
            # serve loop) lets the serve loop pipeline ZMQ recv/decode
            # for this message against the prior round's cache build.
            with torch.profiler.record_function("await_inflight_cache_build"):
                await self._await_inflight_cache_build()

            try:
                with torch.profiler.record_function("handle_speculation_inner"):
                    result = await self._handle_speculation_inner(
                        verify_server_id, identity, outcome
                    )
                if result is not None:
                    cache_hits, draft_tokens, draft_logits, needs_logits = result
                    # Send response FIRST — unblocks the verify server
                    with torch.profiler.record_function("send_speculation_response"):
                        await self._send_speculation_response(
                            verify_server_id,
                            identity,
                            cache_hits,
                            draft_tokens,
                            draft_logits,
                        )
                # Schedule cache build as a background task so the
                # serve loop returns to recv_multipart immediately.
                # The task holds references to the per-round state it
                # needs (seq_ids and B); _await_inflight_cache_build
                # is called at the top of the next SPECULATE so we
                # never have two cache builds running concurrently.
                runner = self.draft_model_runner
                if runner is not None:
                    _seq_ids = self._last_spec_seq_ids
                    if _seq_ids is not None:
                        self._inflight_cache_build = asyncio.create_task(
                            self._run_cache_build(B, _seq_ids, verify_server_id)
                        )
            except Exception:
                logger.exception(
                    "DraftServer _handle_speculation failed for %s",
                    verify_server_id,
                )
                # Send fallback response so the verify server doesn't hang
                try:
                    await self._send_fallback_speculation(verify_server_id, identity, B)
                except Exception:
                    logger.exception(
                        "DraftServer failed to send fallback response to %s",
                        verify_server_id,
                    )

    def _sync_runner_seq_lens_and_blocks(
        self,
        runner: Any,
        seq_ids_list: list[int],
        k_accepted_list: list[int],
    ) -> None:
        """Fix up per-seq KV lengths for this round and reserve headroom.

        ``_seq_lens`` may have been advanced past the accepted position
        by the previous round's JIT or swap. Reconcile from
        ``k_accepted`` relative to the round's base length, then grow
        blocks so there is room for the next JIT or swap to land.
        """
        for i, sid in enumerate(seq_ids_list):
            if sid in self._round_base_lens:
                runner._seq_lens[sid] = (
                    self._round_base_lens[sid] + 1 + int(k_accepted_list[i])
                )

        for sid in seq_ids_list:
            cur_len = runner._seq_lens.get(sid, 0)
            runner.ensure_blocks(sid, cur_len + 2 * self.K + 2)

        # Snapshot base lens BEFORE JIT or swap mutates _seq_lens, so the
        # next round can correct them using this round's k_accepted.
        for sid in seq_ids_list:
            self._round_base_lens[sid] = runner._seq_lens.get(sid, 0)

    def _accumulate_hit_metrics(self, cache_hits: torch.Tensor, batch: int) -> None:
        """Update the rolling cache-hit-rate gauge without a per-round
        sync. Accumulates ``cache_hits.sum()`` lazily into a 0-d GPU
        tensor and only ``.item()``-s it once every
        ``_hit_rate_sync_period`` rounds.
        """
        m = self.metrics
        m._total_lookups += batch
        pending = m._pending_hits_gpu
        if pending is None:
            m._pending_hits_gpu = cache_hits.sum()
        else:
            m._pending_hits_gpu = pending + cache_hits.sum()
        m._hit_rate_sync_round += 1
        if m._hit_rate_sync_round >= m._hit_rate_sync_period:
            m._total_hits += int(m._pending_hits_gpu.item())
            m._pending_hits_gpu = None
            m._hit_rate_sync_round = 0
            if m._total_lookups > 0:
                m.draft_cache_hit_rate.set(m._total_hits / m._total_lookups)

    def _response_for_hits(
        self,
        cache_hits: torch.Tensor,
        cached_tokens: torch.Tensor,
        cached_logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Copy cached tokens/logits into the response buffers.

        Returns (hit_tables, hit_prefix_lens) so the caller can stash
        them for the deferred swap; the synchronous SPECULATE path no
        longer touches runner block tables / seq_lens. None if the
        cache has no hit entries to copy from.

        NOTE: an earlier ``deferred-materialization`` variant tried to
        return ``(None, None)`` sentinels so cache_build could call
        ``get_hit_block_tables`` later in its prologue. That kept the
        SPECULATE hot path sync-free but was unsafe under multi-VS:
        by the time cache_build ran, a peer VS's ``reset_vs`` /
        ``drop_entries_by_seq_ids`` could have compacted the cache
        keys, silently invalidating the stashed match_idx →
        device-side OOB assertion in ``match_idx[hit_mask]``.
        Materializing synchronously here (matching the original ZMQ
        path) makes hit_tables/hit_prefix_lens self-contained tensor
        outputs that survive any subsequent cache mutation.
        """
        hit_tables, hit_prefix_lens = self.cache.get_hit_block_tables(
            cache_hits,
        )
        if hit_tables is None or hit_prefix_lens is None:
            return None
        # torch.where blend of cached tokens/logits into draft buffers,
        # avoiding boolean-mask indexing (which forces a CPU sync via
        # _local_scalar_dense on the mask size).
        K = cached_tokens.shape[1]
        V = cached_logits.shape[2]
        hit_mask_kt = cache_hits.unsqueeze(-1).expand(-1, K)
        hit_mask_ktv = cache_hits.view(-1, 1, 1).expand(-1, K, V)
        draft_tokens.copy_(
            torch.where(hit_mask_kt, cached_tokens, draft_tokens),
        )
        draft_logits.copy_(
            torch.where(hit_mask_ktv, cached_logits, draft_logits),
        )
        return hit_tables, hit_prefix_lens

    def _apply_pending_swap(
        self,
        runner: Any,
        verify_server_id: str,
        seq_ids: torch.Tensor,
        seq_ids_list: list[int],
        cache_hits: torch.Tensor,
        hit_tables: torch.Tensor,
        hit_prefix_lens: torch.Tensor,
    ) -> None:
        """Mutate runner state for a previously-recorded set of cache
        hits. Called at the start of cache_build to hide swap latency
        behind the verifier's target forward.

        ``hit_tables`` / ``hit_prefix_lens`` are always real tensors by
        the time we're called — the IPC hot path stashes None
        sentinels, but ``_run_cache_build`` materializes them via
        ``get_hit_block_tables`` before invoking us (see cache_build.py
        prologue).
        """
        hit_mask = cache_hits.bool()
        hit_seq_ids = seq_ids[hit_mask]
        owned, displaced = runner.swap_block_tables(
            seq_ids=hit_seq_ids,
            branch_block_tables=hit_tables,
            prefix_lens=hit_prefix_lens,
            K=self.K,
        )
        # The hit entries' dedicated blocks were reserved under THIS VS —
        # cache entries for this round's seq_ids can only come from this
        # VS's partition because internal seq_ids are globally unique
        # across VSes.
        for blocks in owned.values():
            runner.exclude_from_dedicated(blocks, verify_server_id)
        if displaced:
            runner._free_list.extend(displaced)
        # Materialize once to avoid 2*H per-element CPU↔GPU syncs.
        hit_indices_list = hit_mask.nonzero(as_tuple=True)[0].tolist()
        hit_prefix_lens_list = hit_prefix_lens.tolist()
        for compact_i, batch_i in enumerate(hit_indices_list):
            sid = seq_ids_list[batch_i]
            runner._seq_lens[sid] = hit_prefix_lens_list[compact_i] + self.K

    def _apply_pending_swap_merged(
        self,
        runner: Any,
        seq_ids: torch.Tensor,
        seq_ids_list: list[int],
        cache_hits: torch.Tensor,
        hit_tables: torch.Tensor,
        hit_prefix_lens: torch.Tensor,
        sid_to_vs: dict[int, str],
    ) -> None:
        """Cross-VS variant of ``_apply_pending_swap``. One merged
        ``swap_block_tables`` call covers all hit seqs; per-VS scoping
        for ``exclude_from_dedicated`` uses the ``sid_to_vs`` map.
        """
        hit_mask = cache_hits.bool()
        hit_seq_ids = seq_ids[hit_mask]
        owned, displaced = runner.swap_block_tables(
            seq_ids=hit_seq_ids,
            branch_block_tables=hit_tables,
            prefix_lens=hit_prefix_lens,
            K=self.K,
        )
        for sid, blocks in owned.items():
            runner.exclude_from_dedicated(
                blocks,
                sid_to_vs.get(sid, "__default__"),
            )
        if displaced:
            runner._free_list.extend(displaced)
        # Materialize once to avoid 2*H per-element CPU↔GPU syncs.
        hit_indices_list = hit_mask.nonzero(as_tuple=True)[0].tolist()
        hit_prefix_lens_list = hit_prefix_lens.tolist()
        for compact_i, batch_i in enumerate(hit_indices_list):
            sid = seq_ids_list[batch_i]
            runner._seq_lens[sid] = hit_prefix_lens_list[compact_i] + self.K

    def _fill_misses_with_zeros(
        self,
        bonus_tokens: torch.Tensor,
        miss_mask: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> None:
        """Write the bonus token into position 0 of every miss row. The
        rest of ``draft_tokens`` / ``draft_logits`` is already zero from
        the caller's ``torch.zeros`` allocation, so we only need to
        overwrite the seed slot. SSD §4.3 fast backup: keep the
        speculate path off the K-step drafter forward. Cache_build
        seeds real cache entries via ``glue_decode(bonus)``.

        Safe to call when ``miss_mask`` is all-False (the masked write
        becomes a no-op) — the caller doesn't need to gate this on a
        host-side ``B_miss`` count.
        """
        # Use torch.where instead of ``x[bool_mask, 0] = y[bool_mask]``.
        # Boolean-mask indexing forces a CPU sync (output size depends
        # on mask), which stalls behind cache_build kernels on the
        # IPC-early-dispatch path.
        col0 = draft_tokens[:, 0]  # [B] view
        draft_tokens[:, 0] = torch.where(miss_mask, bonus_tokens, col0)

    async def _handle_speculation_inner(
        self,
        verify_server_id: str,
        identity: bytes,
        outcome: VerificationOutcome,
        preloaded_tensors: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None] | None
        ) = None,
        preloaded_seq_ids_list: list[int] | None = None,
        preloaded_k_accepted_list: list[int] | None = None,
        ipc_send_ctx: tuple[Any, int, int] | None = None,
    ) -> None:
        """Core speculation logic, separated for error handling.

        ``preloaded_tensors`` is an optional
        ``(seq_ids, k_accepted, bonus_tokens, temperatures)`` tuple.
        When provided (e.g. from the CUDA-IPC transport path) we skip
        the ZMQ frame read; seq_ids must already be remapped into the
        draft server's internal namespace.

        ``preloaded_seq_ids_list`` / ``preloaded_k_accepted_list`` —
        CPU list mirrors of ``seq_ids`` / ``k_accepted``. When provided,
        we skip the corresponding ``.tolist()`` sync (each blocks ~3 ms
        waiting for prior default-stream cache_build kernels). The IPC
        path materializes these via side-stream D2Hs that overlap with
        cache_build.
        """
        B = outcome.batch_size
        _spec_start = time.monotonic()
        self.metrics.draft_batch_size.set(B)

        # ---- Step 1: Receive tensor payloads ----
        with torch.profiler.record_function("recv_spec_tensors"):
            if preloaded_tensors is not None:
                seq_ids, k_accepted, bonus_tokens, temperatures = preloaded_tensors
            else:
                seq_ids, k_accepted, bonus_tokens, temperatures = (
                    self._recv_speculation_tensors(verify_server_id, outcome)
                )
        # Materialize the CPU list first so ``_last_spec_seq_ids`` and
        # ``_last_spec_seq_ids_cpu`` always stay in lock-step (paired
        # assignments after the sync). Downstream cache_build consumers
        # reuse the CPU list to avoid a repeat GPU→host sync at
        # ``_build_next_cache`` — that sync drained the SPECULATE
        # handler's GPU queue for ~1.8 ms per affected iter at 3V c=8.
        if preloaded_seq_ids_list is not None:
            seq_ids_list = preloaded_seq_ids_list
        else:
            seq_ids_list = seq_ids.tolist()
        self._last_spec_seq_ids = seq_ids
        self._last_spec_seq_ids_cpu = seq_ids_list

        # ---- Step 2: Reconcile runner state with this round's base ----
        runner = self.draft_model_runner
        if runner is not None:
            with torch.profiler.record_function("sync_seq_lens"):
                if preloaded_k_accepted_list is not None:
                    k_accepted_list = preloaded_k_accepted_list
                else:
                    k_accepted_list = k_accepted.tolist()
                self._sync_runner_seq_lens_and_blocks(
                    runner,
                    seq_ids_list,
                    k_accepted_list,
                )

        # ---- Step 3: Cache lookup ----
        with torch.profiler.record_function("cache_lookup"):
            cached_tokens, cached_logits, cache_hits, _cached_hs = self.cache.lookup(
                seq_ids=seq_ids,
                k_accepted=k_accepted,
                bonus_tokens=bonus_tokens,
            )

        hit_mask = cache_hits.bool()
        miss_mask = ~hit_mask

        self._accumulate_hit_metrics(cache_hits, B)

        # -------- Phase A: fast IPC response (critical path) --------
        # When ``ipc_send_ctx`` is provided (single-VS IPC handler
        # path), compute JUST cache_hits + draft_tokens and push the
        # response into the peer's ring immediately. Everything else
        # (draft_logits blend, pending_swap swap, _last_* clones)
        # moves to Phase B where it overlaps with the verifier's
        # target forward.
        #
        # cache.lookup already returned cached_tokens[B, K] with zeros
        # on miss rows and real tokens on hit rows. Clone so the
        # bonus-fill write below doesn't alias into the cache buffer.
        draft_tokens = cached_tokens.clone()
        col0 = draft_tokens[:, 0]
        draft_tokens[:, 0] = torch.where(miss_mask, bonus_tokens, col0)

        if ipc_send_ctx is not None:
            peer, slot, seq16 = ipc_send_ctx
            buf = peer.gpu_bufs
            buf["resp_cache_hits"][slot, :B].copy_(
                cache_hits.to(torch.int64),
                non_blocking=True,
            )
            buf["resp_draft_tokens"][slot, :B].copy_(
                draft_tokens,
                non_blocking=True,
            )
            peer.set_resp(slot, seq16)
            self.metrics.draft_generation_latency.observe(
                time.monotonic() - _spec_start
            )
            # Hand Phase B off asynchronously via
            # ``_inflight_cache_build``. Serve loop can return to
            # polling doorbells while draft_logits blend, pending_swap
            # setup, _last_* clones, and cache_build all run in the
            # background. Next round's SPECULATE awaits via
            # ``_await_inflight_cache_build`` before mutating shared
            # runner state, so the deferred ``_pending_swap`` /
            # ``_last_*`` writes remain safe.
            self._inflight_cache_build = asyncio.create_task(
                self._phase_b_and_cache_build_solo(
                    {
                        "verify_server_id": verify_server_id,
                        "B": B,
                        "cache_hits": cache_hits,
                        "cached_logits": cached_logits,
                        "hit_mask": hit_mask,
                        "miss_mask": miss_mask,
                        "draft_tokens": draft_tokens,
                        "seq_ids": seq_ids,
                        "seq_ids_list": seq_ids_list,
                        "bonus_tokens": bonus_tokens,
                        "runner": runner,
                    }
                )
            )
            return None

        # ZMQ / non-IPC path: original synchronous Phase B + return.
        draft_logits = torch.zeros(
            B,
            self.K,
            self.vocab_size,
            dtype=self.dtype,
            device=self.device,
        )

        # ---- Step 4: Copy cache hits into response (swap deferred) ----
        used_swap_for_hits = False
        pending_swap: dict[str, Any] | None = None
        if cached_logits is not None and runner is not None:
            with torch.profiler.record_function("copy_hits"):
                hit_tables, hit_prefix_lens = self.cache.get_hit_block_tables(
                    cache_hits
                )
                if hit_tables is not None and hit_prefix_lens is not None:
                    hit_mask_ktv = cache_hits.view(-1, 1, 1).expand(
                        -1,
                        self.K,
                        self.vocab_size,
                    )
                    draft_logits.copy_(
                        torch.where(
                            hit_mask_ktv,
                            cached_logits,
                            draft_logits,
                        ),
                    )
                    pending_swap = {
                        "verify_server_id": verify_server_id,
                        "seq_ids": seq_ids,
                        "seq_ids_list": seq_ids_list,
                        "cache_hits": cache_hits,
                        "hit_tables": hit_tables,
                        "hit_prefix_lens": hit_prefix_lens,
                    }
                    used_swap_for_hits = True
        self._pending_swap = pending_swap

        if not used_swap_for_hits:
            for sid in seq_ids_list:
                self._swap_states[sid] = {}

        # Stash for _build_next_cache
        self._last_draft_tokens = draft_tokens.clone()
        self._last_draft_logits = draft_logits.clone()
        self._last_bonus_tokens = bonus_tokens.clone()
        self._last_miss_mask = miss_mask.clone()

        send_logits = outcome.needs_logits
        self.metrics.draft_generation_latency.observe(time.monotonic() - _spec_start)
        return (
            cache_hits,
            draft_tokens,
            draft_logits if send_logits else None,
            send_logits,
        )

    # ------------------------------------------------------------------
    # Merged SPECULATE handler (Option A: cross-VS batching)
    # ------------------------------------------------------------------

    async def _handle_speculation_merged(
        self,
        items: list[tuple[str, bytes, DraftCommand, list[bytes]]],
    ) -> None:
        """Run two SPECULATEs (one per VS) as a single merged forward.

        Each item is (vs_id, identity, command, tensor_frames). The
        merge concatenates seq_ids/k_accepted/bonus/temperatures along
        the batch dim, runs ONE pass through cache_lookup → swap →
        jit → response, then splits outputs back per-VS, sends
        per-VS SpeculationResponse, and schedules per-VS cache builds.

        Per-VS bookkeeping (dedicated-block exclusion, swap_states,
        round_base_lens, _last_*_tokens for cache build) stays
        per-VS-correct because we track which batch indices originated
        from which VS via ``entry_vs``.
        """
        with torch.profiler.record_function(f"speculate_merged_n{len(items)}"):
            with torch.profiler.record_function("await_inflight_cache_build"):
                await self._await_inflight_cache_build()

            try:
                await self._handle_speculation_merged_inner(items)
            except Exception:
                logger.exception(
                    "DraftServer _handle_speculation_merged failed; "
                    "falling back to per-VS error responses."
                )
                for vs_id, identity, command, _frames in items:
                    outcome = decode(command.payload, VerificationOutcome)
                    try:
                        await self._send_fallback_speculation(
                            vs_id,
                            identity,
                            outcome.batch_size,
                        )
                    except Exception:
                        logger.exception(
                            "DraftServer fallback to %s failed",
                            vs_id,
                        )

    async def _handle_speculation_merged_inner(
        self,
        items: list[tuple[str, bytes, DraftCommand, list[bytes]]],
        preloaded_per_vs: list[dict[str, Any]] | None = None,
    ) -> None:
        """Merged handler shared by the ZMQ and CUDA-IPC paths.

        When ``preloaded_per_vs`` is None (ZMQ) each item's tensors are
        decoded from its ZMQ frames via ``_recv_speculation_tensors``.

        When ``preloaded_per_vs`` is provided (IPC) it's a list of
        already-materialized per-VS dicts with keys
        ``vs_id / identity / outcome / B / seq_ids / k_accepted /
        bonus_tokens / temperatures / seq_ids_list / k_accepted_list``
        — same shape the ZMQ path constructs below. The caller
        (``_handle_ipc_speculation_merged``) provides both the GPU
        tensors and the pre-materialized CPU lists so we skip the
        ``.tolist()`` sync at line ``seq_ids_list = seq_ids_cat.tolist()``.
        """
        # Each merged item is a real SPECULATE that, in the unmerged
        # path, would have incremented draft_speculate_total via
        # _dispatch. Mirror that here, plus mark how many participated
        # in a merge (for counting merge effectiveness).
        self.metrics.draft_speculate_total.inc(len(items))
        if len(items) >= 2:
            self.metrics.draft_speculate_merged.inc(len(items))

        runner = self.draft_model_runner
        if runner is None:
            if preloaded_per_vs is not None:
                # IPC path: fallback is per-peer zero-response fill.
                for p in preloaded_per_vs:
                    if "ipc_peer" not in p:
                        continue
                    peer = p["ipc_peer"]
                    slot = p["ipc_slot"]
                    seq16 = p["ipc_seq16"]
                    B = p["B"]
                    buf = peer.gpu_bufs
                    buf["resp_cache_hits"][slot, :B].zero_()
                    buf["resp_draft_tokens"][slot, :B].zero_()
                    peer.set_resp(slot, seq16)
                return
            for vs_id, identity, command, _ in items:
                outcome = decode(command.payload, VerificationOutcome)
                await self._send_fallback_speculation(
                    vs_id,
                    identity,
                    outcome.batch_size,
                )
            return

        # ---- Per-VS recv: read each VS's tensors ----
        if preloaded_per_vs is not None:
            per_vs = preloaded_per_vs
        else:
            per_vs = []
            for vs_id, identity, command, frames in items:
                outcome = decode(command.payload, VerificationOutcome)
                seq_ids, k_accepted, bonus_tokens, temperatures = (
                    self._recv_speculation_tensors(
                        vs_id,
                        outcome,
                        frames=frames,
                    )
                )
                per_vs.append(
                    {
                        "vs_id": vs_id,
                        "identity": identity,
                        "outcome": outcome,
                        "B": outcome.batch_size,
                        "seq_ids": seq_ids,
                        "k_accepted": k_accepted,
                        "bonus_tokens": bonus_tokens,
                        "temperatures": temperatures,
                    }
                )

        # ---- Concatenate along batch dim ----
        seq_ids_cat = torch.cat([p["seq_ids"] for p in per_vs], dim=0)
        k_accepted_cat = torch.cat([p["k_accepted"] for p in per_vs], dim=0)
        bonus_cat = torch.cat([p["bonus_tokens"] for p in per_vs], dim=0)

        # entry_vs[i] = index into items for the i-th merged batch row
        entry_vs: list[int] = []
        for vs_idx, p in enumerate(per_vs):
            entry_vs.extend([vs_idx] * p["B"])

        B_total = seq_ids_cat.shape[0]
        # IPC path pre-materialized CPU lists per VS via side-stream
        # D2Hs; concatenate them here to skip the ``.tolist()`` syncs
        # that would otherwise stall behind cache_build kernels on the
        # default stream.
        if preloaded_per_vs is not None and all("seq_ids_list" in p for p in per_vs):
            seq_ids_list = [sid for p in per_vs for sid in p["seq_ids_list"]]
        else:
            seq_ids_list = seq_ids_cat.tolist()
        if preloaded_per_vs is not None and all("k_accepted_list" in p for p in per_vs):
            k_accepted_list = [ka for p in per_vs for ka in p["k_accepted_list"]]
        else:
            k_accepted_list = k_accepted_cat.tolist()

        self._last_spec_seq_ids = seq_ids_cat
        # Mirror the single-VS path: stash the CPU list for downstream
        # cache_build consumers that would otherwise re-tolist().
        self._last_spec_seq_ids_cpu = seq_ids_list
        self.metrics.draft_batch_size.set(B_total)
        _spec_start = time.monotonic()

        # ---- Reconcile runner state ----
        with torch.profiler.record_function("sync_seq_lens_merged"):
            self._sync_runner_seq_lens_and_blocks(
                runner,
                seq_ids_list,
                k_accepted_list,
            )

        # ---- Cache lookup (one merged call) ----
        with torch.profiler.record_function("cache_lookup_merged"):
            cached_tokens, cached_logits, cache_hits, _hs = self.cache.lookup(
                seq_ids=seq_ids_cat,
                k_accepted=k_accepted_cat,
                bonus_tokens=bonus_cat,
            )
        hit_mask = cache_hits.bool()
        miss_mask = ~hit_mask
        self._accumulate_hit_metrics(cache_hits, B_total)

        # -------- Phase A: fast IPC response (critical path) --------
        # The IPC response wire only carries cache_hits + draft_tokens
        # (no logits — needs_logits=True falls back to ZMQ upstream).
        # Compute ONLY those two tensors on the critical path so we can
        # signal the verifier immediately, then move all the heavier
        # per-round staging (draft_logits blend, _pending_swap_merged
        # build, _last_* clones, get_hit_block_tables) to Phase B below
        # where it overlaps with the verifier's target forward.
        all_ipc = all("ipc_peer" in p and not p["outcome"].needs_logits for p in per_vs)
        # cache.lookup already returned cached_tokens[B, K] with zeros
        # on miss rows and real tokens on hit rows. Clone so downstream
        # miss-fill can mutate col 0 without affecting the cache. Cost:
        # B×K×i64 = a few KB, no allocator pressure.
        draft_tokens = cached_tokens.clone()
        # Zero-fallback seed: bonus token in col 0 for miss rows.
        col0 = draft_tokens[:, 0]
        draft_tokens[:, 0] = torch.where(miss_mask, bonus_cat, col0)

        if all_ipc:
            # Send responses NOW, before doing any of the cache_build
            # prep work. Each peer sees ~2 D2D copies + 1 doorbell
            # kernel-queued write; the verifier unblocks in ~1-2 ms
            # instead of waiting for the 10-12 ms of prep that used to
            # come first.
            offset = 0
            for vs_idx, p in enumerate(per_vs):
                B = p["B"]
                sl = slice(offset, offset + B)
                peer = p["ipc_peer"]
                slot = p["ipc_slot"]
                seq16 = p["ipc_seq16"]
                buf = peer.gpu_bufs
                buf["resp_cache_hits"][slot, :B].copy_(
                    cache_hits[sl].to(torch.int64),
                    non_blocking=True,
                )
                buf["resp_draft_tokens"][slot, :B].copy_(
                    draft_tokens[sl],
                    non_blocking=True,
                )
                peer.set_resp(slot, seq16)
                offset += B
            self.metrics.draft_generation_latency.observe(
                time.monotonic() - _spec_start
            )

            # -------- Async Phase B + cache_build ----------------
            # Everything after the response send (draft_logits blend,
            # get_hit_block_tables, _apply_pending_swap_merged,
            # _last_* clones, slice_metas construction) becomes part
            # of the ``_inflight_cache_build`` task so the serve loop
            # can return to polling doorbells immediately after the
            # response D2D + doorbell fill are kernel-queued. Next
            # round's SPECULATE awaits this task via
            # ``_await_inflight_cache_build`` before mutating cache
            # state; runner state mutations from
            # ``_apply_pending_swap_merged`` are therefore visible
            # before ``_sync_runner_seq_lens_and_blocks`` next round.
            phase_b_ctx = {
                "cache_hits": cache_hits,
                "cached_logits": cached_logits,
                "hit_mask": hit_mask,
                "miss_mask": miss_mask,
                "draft_tokens": draft_tokens,
                "seq_ids_cat": seq_ids_cat,
                "seq_ids_list": seq_ids_list,
                "bonus_cat": bonus_cat,
                "per_vs": per_vs,
                "entry_vs": entry_vs,
                "B_total": B_total,
                "runner": runner,
            }
            self._inflight_cache_build = asyncio.create_task(
                self._phase_b_and_cache_build_merged(phase_b_ctx)
            )
            return

        # ZMQ (or mixed transport) path: original synchronous Phase B
        # + response send, kept for ZMQ callers that need
        # ``draft_logits`` in the response.
        draft_logits = torch.zeros(
            B_total,
            self.K,
            self.vocab_size,
            dtype=self.dtype,
            device=self.device,
        )

        # ---- Apply cache hits (swap inline on merged path) ----
        used_swap_for_hits = False
        self._pending_swap_merged = None
        if cached_logits is not None:
            with torch.profiler.record_function("swap_hits_merged"):
                hit_tables, hit_prefix_lens = self.cache.get_hit_block_tables(
                    cache_hits
                )
                if hit_tables is not None and hit_prefix_lens is not None:
                    draft_logits[hit_mask] = cached_logits[hit_mask]
                    sid_to_vs: dict[int, str] = {}
                    for i, sid in enumerate(seq_ids_list):
                        sid_to_vs[sid] = per_vs[entry_vs[i]]["vs_id"]
                    self._apply_pending_swap_merged(
                        runner=runner,
                        seq_ids=seq_ids_cat,
                        seq_ids_list=seq_ids_list,
                        cache_hits=cache_hits,
                        hit_tables=hit_tables,
                        hit_prefix_lens=hit_prefix_lens,
                        sid_to_vs=sid_to_vs,
                    )
                    self._pending_swap_merged = {
                        "seq_ids": seq_ids_cat,
                        "seq_ids_list": seq_ids_list,
                        "cache_hits": cache_hits,
                        "hit_tables": hit_tables,
                        "hit_prefix_lens": hit_prefix_lens,
                        "sid_to_vs": sid_to_vs,
                    }
                    used_swap_for_hits = True

        if not used_swap_for_hits:
            for sid in seq_ids_list:
                self._swap_states[sid] = {}

        # Stash for cache build (split per-VS below).
        self._last_draft_tokens = draft_tokens.clone()
        self._last_draft_logits = draft_logits.clone()
        self._last_bonus_tokens = bonus_cat.clone()
        self._last_miss_mask = miss_mask.clone()

        self.metrics.draft_generation_latency.observe(time.monotonic() - _spec_start)

        # ---- Send per-VS responses (ZMQ path) ----
        offset = 0
        for vs_idx, p in enumerate(per_vs):
            B = p["B"]
            sl = slice(offset, offset + B)
            send_logits = p["outcome"].needs_logits
            with torch.profiler.record_function("send_speculation_response"):
                await self._send_speculation_response(
                    p["vs_id"],
                    p["identity"],
                    cache_hits[sl],
                    draft_tokens[sl],
                    draft_logits[sl] if send_logits else None,
                )
            offset += B

        # ---- Schedule one merged cache build covering both VSes ----
        slice_metas: list[dict[str, Any]] = []
        offset = 0
        for vs_idx, p in enumerate(per_vs):
            B = p["B"]
            sl = slice(offset, offset + B)
            sm = {
                "vs_id": p["vs_id"],
                "B": B,
                "seq_ids": seq_ids_cat[sl].clone(),
                "seq_ids_cpu": seq_ids_list[offset : offset + B],
                "bonus_tokens": bonus_cat[sl].clone(),
                "draft_tokens": draft_tokens[sl].clone(),
                "draft_logits": draft_logits[sl].clone(),
            }
            if self._last_miss_mask is not None:
                sm["miss_mask"] = self._last_miss_mask[sl].clone()
            slice_metas.append(sm)
            offset += B
        self._inflight_cache_build = asyncio.create_task(
            self._run_cache_build_merged(slice_metas)
        )

    async def _phase_b_and_cache_build_solo(
        self,
        ctx: dict[str, Any],
    ) -> None:
        """Async continuation of the single-VS IPC path. Runs Phase B
        (draft_logits blend + pending_swap setup + _last_* clones),
        then invokes ``_run_cache_build``. All-miss shortcut: when
        ``cached_logits`` is None or ``get_hit_block_tables`` returns
        None, skip all the hit-side work and go straight to cache_build.
        """
        verify_server_id = ctx["verify_server_id"]
        B = ctx["B"]
        cache_hits = ctx["cache_hits"]
        cached_logits = ctx["cached_logits"]
        miss_mask = ctx["miss_mask"]
        draft_tokens = ctx["draft_tokens"]
        seq_ids = ctx["seq_ids"]
        seq_ids_list = ctx["seq_ids_list"]
        bonus_tokens = ctx["bonus_tokens"]
        runner = ctx["runner"]

        draft_logits = torch.zeros(
            B,
            self.K,
            self.vocab_size,
            dtype=self.dtype,
            device=self.device,
        )

        used_swap_for_hits = False
        pending_swap: dict[str, Any] | None = None
        if cached_logits is not None and runner is not None:
            with torch.profiler.record_function("copy_hits"):
                hit_tables, hit_prefix_lens = self.cache.get_hit_block_tables(
                    cache_hits
                )
                if hit_tables is not None and hit_prefix_lens is not None:
                    hit_mask_ktv = cache_hits.view(-1, 1, 1).expand(
                        -1,
                        self.K,
                        self.vocab_size,
                    )
                    draft_logits.copy_(
                        torch.where(
                            hit_mask_ktv,
                            cached_logits,
                            draft_logits,
                        ),
                    )
                    pending_swap = {
                        "verify_server_id": verify_server_id,
                        "seq_ids": seq_ids,
                        "seq_ids_list": seq_ids_list,
                        "cache_hits": cache_hits,
                        "hit_tables": hit_tables,
                        "hit_prefix_lens": hit_prefix_lens,
                    }
                    used_swap_for_hits = True
        self._pending_swap = pending_swap

        if not used_swap_for_hits:
            for sid in seq_ids_list:
                self._swap_states[sid] = {}

        # Stash for _build_next_cache
        self._last_draft_tokens = draft_tokens.clone()
        self._last_draft_logits = draft_logits.clone()
        self._last_bonus_tokens = bonus_tokens.clone()
        self._last_miss_mask = miss_mask.clone()
        self._last_spec_seq_ids = seq_ids
        self._last_spec_seq_ids_cpu = seq_ids_list

        # Now the actual cache build. Awaited inline because we're
        # already the task the caller scheduled.
        await self._run_cache_build(B, seq_ids, verify_server_id)

    async def _phase_b_and_cache_build_merged(
        self,
        ctx: dict[str, Any],
    ) -> None:
        """Async continuation of the IPC merged path.

        Runs Phase B (get_hit_block_tables, _apply_pending_swap_merged,
        draft_logits blend, _last_* clones, slice_metas build) and
        then invokes ``_run_cache_build_merged``. The whole thing is
        scheduled as ``_inflight_cache_build`` so the serve loop can
        return to polling doorbells immediately after Phase A signals
        the verifiers. Next round's SPECULATE awaits this task via
        ``_await_inflight_cache_build`` before touching runner state,
        so ``_apply_pending_swap_merged``'s mutations of
        ``runner._block_table_gpu`` / ``runner._seq_lens`` are
        guaranteed visible before ``_sync_runner_seq_lens_and_blocks``
        reads them next round.

        All-miss fast path: when ``cache_hits`` is all-False,
        ``get_hit_block_tables`` returns ``(None, None)`` and there's
        no swap to apply and no cached logits to blend. Skip the
        get_hit_block_tables call entirely to avoid its host-side
        ``.item()`` sync and go straight to the ``_last_*`` stashing
        needed by cache_build's cleanup_glue path.
        """
        cache_hits = ctx["cache_hits"]
        cached_logits = ctx["cached_logits"]
        hit_mask = ctx["hit_mask"]
        miss_mask = ctx["miss_mask"]
        draft_tokens = ctx["draft_tokens"]
        seq_ids_cat = ctx["seq_ids_cat"]
        seq_ids_list = ctx["seq_ids_list"]
        bonus_cat = ctx["bonus_cat"]
        per_vs = ctx["per_vs"]
        entry_vs = ctx["entry_vs"]
        B_total = ctx["B_total"]
        runner = ctx["runner"]

        draft_logits = torch.zeros(
            B_total,
            self.K,
            self.vocab_size,
            dtype=self.dtype,
            device=self.device,
        )

        # All-miss shortcut: no swap, no blend, no pending_swap_merged.
        # ``cached_logits is None`` when the cache was completely empty
        # (first round, or every VS had reset). ``get_hit_block_tables``
        # returns (None, None) when num_hits == 0.
        used_swap_for_hits = False
        self._pending_swap_merged = None
        if cached_logits is not None:
            with torch.profiler.record_function("swap_hits_merged"):
                hit_tables, hit_prefix_lens = self.cache.get_hit_block_tables(
                    cache_hits
                )
                if hit_tables is not None and hit_prefix_lens is not None:
                    draft_logits[hit_mask] = cached_logits[hit_mask]
                    sid_to_vs: dict[int, str] = {}
                    for i, sid in enumerate(seq_ids_list):
                        sid_to_vs[sid] = per_vs[entry_vs[i]]["vs_id"]
                    self._apply_pending_swap_merged(
                        runner=runner,
                        seq_ids=seq_ids_cat,
                        seq_ids_list=seq_ids_list,
                        cache_hits=cache_hits,
                        hit_tables=hit_tables,
                        hit_prefix_lens=hit_prefix_lens,
                        sid_to_vs=sid_to_vs,
                    )
                    self._pending_swap_merged = {
                        "seq_ids": seq_ids_cat,
                        "seq_ids_list": seq_ids_list,
                        "cache_hits": cache_hits,
                        "hit_tables": hit_tables,
                        "hit_prefix_lens": hit_prefix_lens,
                        "sid_to_vs": sid_to_vs,
                    }
                    used_swap_for_hits = True

        if not used_swap_for_hits:
            for sid in seq_ids_list:
                self._swap_states[sid] = {}

        # Stash for cache build.
        self._last_draft_tokens = draft_tokens.clone()
        self._last_draft_logits = draft_logits.clone()
        self._last_bonus_tokens = bonus_cat.clone()
        self._last_miss_mask = miss_mask.clone()

        # Build per-VS slice metas and run cache_build inline (we're
        # already in the cache_build task, so this is just a call).
        slice_metas: list[dict[str, Any]] = []
        offset = 0
        for vs_idx, p in enumerate(per_vs):
            B = p["B"]
            sl = slice(offset, offset + B)
            sm = {
                "vs_id": p["vs_id"],
                "B": B,
                "seq_ids": seq_ids_cat[sl].clone(),
                "seq_ids_cpu": seq_ids_list[offset : offset + B],
                "bonus_tokens": bonus_cat[sl].clone(),
                "draft_tokens": draft_tokens[sl].clone(),
                "draft_logits": draft_logits[sl].clone(),
            }
            if self._last_miss_mask is not None:
                sm["miss_mask"] = self._last_miss_mask[sl].clone()
            slice_metas.append(sm)
            offset += B
        await self._run_cache_build_merged(slice_metas)
