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
                            verify_server_id, identity, cache_hits, draft_tokens,
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
                            self._run_cache_build(
                                B, _seq_ids, verify_server_id
                            )
                        )
            except Exception:
                logger.exception(
                    "DraftServer _handle_speculation failed for %s",
                    verify_server_id,
                )
                # Send fallback response so the verify server doesn't hang
                try:
                    await self._send_fallback_speculation(
                        verify_server_id, identity, B
                    )
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
                    self._round_base_lens[sid]
                    + 1
                    + int(k_accepted_list[i])
                )

        for sid in seq_ids_list:
            cur_len = runner._seq_lens.get(sid, 0)
            runner.ensure_blocks(sid, cur_len + 2 * self.K + 2)

        # Snapshot base lens BEFORE JIT or swap mutates _seq_lens, so the
        # next round can correct them using this round's k_accepted.
        for sid in seq_ids_list:
            self._round_base_lens[sid] = runner._seq_lens.get(sid, 0)

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
        longer touches runner block tables / seq_lens. None if the cache
        has no hit entries to copy from.
        """
        hit_tables, hit_prefix_lens = self.cache.get_hit_block_tables(
            cache_hits
        )
        if hit_tables is None or hit_prefix_lens is None:
            return None
        hit_mask = cache_hits.bool()
        draft_tokens[hit_mask] = cached_tokens[hit_mask]
        draft_logits[hit_mask] = cached_logits[hit_mask]
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
                blocks, sid_to_vs.get(sid, "__default__"),
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
        B_miss: int,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> None:
        """Write zero drafts (with bonus at position 0) into the miss
        subset of ``draft_tokens`` / ``draft_logits``. SSD §4.3 fast
        backup: keep the speculate path off the K-step drafter forward.
        Cache_build seeds real cache entries via ``glue_decode(bonus)``.
        """
        miss_bonus = bonus_tokens[miss_mask]
        zero_tokens, zero_logits = self._zero_drafts_for_misses(
            miss_bonus, B_miss=B_miss,
        )
        draft_tokens[miss_mask] = zero_tokens
        draft_logits[miss_mask] = zero_logits

    async def _handle_speculation_inner(
        self,
        verify_server_id: str,
        identity: bytes,
        outcome: VerificationOutcome,
    ) -> None:
        """Core speculation logic, separated for error handling."""
        B = outcome.batch_size
        _spec_start = time.monotonic()
        self.metrics.draft_batch_size.set(B)

        # ---- Step 1: Receive tensor payloads ----
        with torch.profiler.record_function("recv_spec_tensors"):
            seq_ids, k_accepted, bonus_tokens, temperatures = (
                self._recv_speculation_tensors(verify_server_id, outcome)
            )
        self._last_spec_seq_ids = seq_ids
        seq_ids_list = seq_ids.tolist()

        # ---- Step 2: Reconcile runner state with this round's base ----
        runner = self.draft_model_runner
        if runner is not None:
            with torch.profiler.record_function("sync_seq_lens"):
                self._sync_runner_seq_lens_and_blocks(
                    runner, seq_ids_list, k_accepted.tolist(),
                )

        # ---- Step 3: Cache lookup ----
        with torch.profiler.record_function("cache_lookup"):
            cached_tokens, cached_logits, cache_hits, _cached_hs = (
                self.cache.lookup(
                    seq_ids=seq_ids,
                    k_accepted=k_accepted,
                    bonus_tokens=bonus_tokens,
                )
            )

        num_hits = int(cache_hits.sum().item())
        hit_mask = cache_hits.bool()
        miss_mask = ~hit_mask

        self.metrics._total_lookups += B
        self.metrics._total_hits += num_hits
        if self.metrics._total_lookups > 0:
            self.metrics.draft_cache_hit_rate.set(
                self.metrics._total_hits / self.metrics._total_lookups
            )

        draft_tokens = torch.zeros(
            B, self.K, dtype=torch.int64, device=self.device,
        )
        draft_logits = torch.zeros(
            B, self.K, self.vocab_size,
            dtype=self.dtype, device=self.device,
        )

        # ---- Step 4: Copy cache hits into response (swap deferred) ----
        # The runner-state mutation (swap_block_tables, _seq_lens,
        # _free_list) runs at the top of _run_cache_build instead of
        # here, so the synchronous SPECULATE path returns ~2 ms sooner.
        # Cache_build awaits the verifier's target forward anyway, so
        # the deferred swap hides behind that overlap.
        used_swap_for_hits = False
        pending_swap: dict[str, Any] | None = None
        if num_hits > 0 and cached_logits is not None and runner is not None:
            with torch.profiler.record_function(f"copy_hits_{num_hits}"):
                hit_payload = self._response_for_hits(
                    cache_hits=cache_hits,
                    cached_tokens=cached_tokens,
                    cached_logits=cached_logits,
                    draft_tokens=draft_tokens,
                    draft_logits=draft_logits,
                )
                if hit_payload is not None:
                    hit_tables, hit_prefix_lens = hit_payload
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

        # ---- Step 5: zero-fill on misses ----
        B_miss = int(miss_mask.sum().item())
        if B_miss > 0:
            with torch.profiler.record_function(f"miss_fill_B{B_miss}"):
                self._fill_misses_with_zeros(
                    bonus_tokens=bonus_tokens,
                    miss_mask=miss_mask,
                    B_miss=B_miss,
                    draft_tokens=draft_tokens,
                    draft_logits=draft_logits,
                )

        if not used_swap_for_hits:
            for sid in seq_ids_list:
                self._swap_states[sid] = {}

        # Stash for _build_next_cache
        self._last_draft_tokens = draft_tokens.clone()
        self._last_draft_logits = draft_logits.clone()
        self._last_bonus_tokens = bonus_tokens.clone()
        # Tell cache_build which rows had zero-dummy drafts so it
        # uses bonus_tokens (not draft_tokens[:,-1]) for glue_decode
        # and seeds branches from base+1 instead of base+K.
        self._last_miss_mask = miss_mask.clone()

        send_logits = outcome.needs_logits
        self.metrics.draft_generation_latency.observe(
            time.monotonic() - _spec_start
        )
        return (cache_hits, draft_tokens,
                draft_logits if send_logits else None,
                send_logits)

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
        with torch.profiler.record_function(
            f"speculate_merged_n{len(items)}"
        ):
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
                            vs_id, identity, outcome.batch_size,
                        )
                    except Exception:
                        logger.exception(
                            "DraftServer fallback to %s failed", vs_id,
                        )

    async def _handle_speculation_merged_inner(
        self,
        items: list[tuple[str, bytes, DraftCommand, list[bytes]]],
    ) -> None:
        # Each merged item is a real SPECULATE that, in the unmerged
        # path, would have incremented draft_speculate_total via
        # _dispatch. Mirror that here, plus mark how many participated
        # in a merge (for counting merge effectiveness).
        self.metrics.draft_speculate_total.inc(len(items))
        if len(items) >= 2:
            self.metrics.draft_speculate_merged.inc(len(items))

        runner = self.draft_model_runner
        if runner is None:
            for vs_id, identity, command, _ in items:
                outcome = decode(command.payload, VerificationOutcome)
                await self._send_fallback_speculation(
                    vs_id, identity, outcome.batch_size,
                )
            return

        # ---- Per-VS recv: read each VS's tensors from its own frames ----
        per_vs: list[dict[str, Any]] = []
        for vs_id, identity, command, frames in items:
            outcome = decode(command.payload, VerificationOutcome)
            seq_ids, k_accepted, bonus_tokens, temperatures = (
                self._recv_speculation_tensors(
                    vs_id, outcome, frames=frames,
                )
            )
            per_vs.append({
                "vs_id": vs_id,
                "identity": identity,
                "outcome": outcome,
                "B": outcome.batch_size,
                "seq_ids": seq_ids,
                "k_accepted": k_accepted,
                "bonus_tokens": bonus_tokens,
                "temperatures": temperatures,
            })

        # ---- Concatenate along batch dim ----
        seq_ids_cat = torch.cat([p["seq_ids"] for p in per_vs], dim=0)
        k_accepted_cat = torch.cat([p["k_accepted"] for p in per_vs], dim=0)
        bonus_cat = torch.cat([p["bonus_tokens"] for p in per_vs], dim=0)
        # All-or-nothing on temperatures: if any VS sent them, all must.
        if all(p["temperatures"] is not None for p in per_vs):
            temps_cat: torch.Tensor | None = torch.cat(
                [p["temperatures"] for p in per_vs], dim=0,
            )
        else:
            temps_cat = None

        # entry_vs[i] = index into items for the i-th merged batch row
        entry_vs: list[int] = []
        for vs_idx, p in enumerate(per_vs):
            entry_vs.extend([vs_idx] * p["B"])

        B_total = seq_ids_cat.shape[0]
        seq_ids_list = seq_ids_cat.tolist()

        self._last_spec_seq_ids = seq_ids_cat
        self.metrics.draft_batch_size.set(B_total)
        _spec_start = time.monotonic()

        # ---- Reconcile runner state ----
        with torch.profiler.record_function("sync_seq_lens_merged"):
            self._sync_runner_seq_lens_and_blocks(
                runner, seq_ids_list, k_accepted_cat.tolist(),
            )

        # ---- Cache lookup (one merged call) ----
        with torch.profiler.record_function("cache_lookup_merged"):
            cached_tokens, cached_logits, cache_hits, _hs = (
                self.cache.lookup(
                    seq_ids=seq_ids_cat,
                    k_accepted=k_accepted_cat,
                    bonus_tokens=bonus_cat,
                )
            )
        num_hits = int(cache_hits.sum().item())
        hit_mask = cache_hits.bool()
        miss_mask = ~hit_mask
        self.metrics._total_lookups += B_total
        self.metrics._total_hits += num_hits
        if self.metrics._total_lookups > 0:
            self.metrics.draft_cache_hit_rate.set(
                self.metrics._total_hits / self.metrics._total_lookups
            )

        draft_tokens = torch.zeros(
            B_total, self.K, dtype=torch.int64, device=self.device,
        )
        draft_logits = torch.zeros(
            B_total, self.K, self.vocab_size,
            dtype=self.dtype, device=self.device,
        )

        # ---- Apply cache hits (swap inline on merged path) ----
        # Multi-VS deployments showed a small TPOT/ITL regression when
        # the swap was deferred to cache_build's prologue (2V/3V at c=8
        # tracked +1-3 % at 1B sequential), because the cross-VS merge
        # peek window leaves less slack for the next round to absorb
        # cache_build prologue work. Keep swap synchronous on the merged
        # path; the single-VS path still defers via _pending_swap.
        # Hit metadata is still stashed in _pending_swap_merged for the
        # parallel-fanout KV cleanup step at the top of cache_build.
        used_swap_for_hits = False
        self._pending_swap_merged = None
        if num_hits > 0 and cached_logits is not None:
            with torch.profiler.record_function(
                f"swap_hits_merged_{num_hits}"
            ):
                hit_tables, hit_prefix_lens = (
                    self.cache.get_hit_block_tables(cache_hits)
                )
                if hit_tables is not None and hit_prefix_lens is not None:
                    draft_tokens[hit_mask] = cached_tokens[hit_mask]
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

        # ---- zero-fill on misses (one merged call) ----
        B_miss = int(miss_mask.sum().item())
        if B_miss > 0:
            with torch.profiler.record_function(
                f"miss_fill_merged_B{B_miss}"
            ):
                self._fill_misses_with_zeros(
                    bonus_tokens=bonus_cat,
                    miss_mask=miss_mask,
                    B_miss=B_miss,
                    draft_tokens=draft_tokens,
                    draft_logits=draft_logits,
                )

        if not used_swap_for_hits:
            for sid in seq_ids_list:
                self._swap_states[sid] = {}

        # Stash for cache build (split per-VS below).
        self._last_draft_tokens = draft_tokens.clone()
        self._last_draft_logits = draft_logits.clone()
        self._last_bonus_tokens = bonus_cat.clone()
        self._last_miss_mask = miss_mask.clone()

        self.metrics.draft_generation_latency.observe(
            time.monotonic() - _spec_start
        )

        # ---- Send per-VS responses ----
        offset = 0
        for vs_idx, p in enumerate(per_vs):
            B = p["B"]
            sl = slice(offset, offset + B)
            send_logits = p["outcome"].needs_logits
            with torch.profiler.record_function("send_speculation_response"):
                await self._send_speculation_response(
                    p["vs_id"], p["identity"],
                    cache_hits[sl], draft_tokens[sl],
                    draft_logits[sl] if send_logits else None,
                )
            offset += B

        # ---- Schedule one merged cache build covering both VSes ----
        # Cache build's per-VS scoping is only needed for cache_partition
        # reset and dedicated-block ownership; the actual GPU work
        # (glue_decode, allocate-and-copy-KV, tree_decode) is naturally
        # batched and runs ~once per merged round instead of twice.
        slice_metas: list[dict[str, Any]] = []
        offset = 0
        for vs_idx, p in enumerate(per_vs):
            B = p["B"]
            sl = slice(offset, offset + B)
            sm = {
                "vs_id": p["vs_id"],
                "B": B,
                "seq_ids": seq_ids_cat[sl].clone(),
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


    # ------------------------------------------------------------------
    # Zero-fallback (cache miss path)
    # ------------------------------------------------------------------

    def _zero_drafts_for_misses(
        self,
        bonus_tokens: torch.Tensor,
        B_miss: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero drafts (with bonus at position 0) for cache-miss
        rows — SSD §4.3 fast backup. Saves the K-step drafter forward
        on the speculate critical path; cache_build runs glue_decode
        on the bonus token to seed cache entries for the next round.
        Caller marks ``self._last_miss_mask`` so cache_build knows
        which rows need bonus-token glue_decode (instead of drafted-
        token glue_decode).
        """
        with torch.profiler.record_function(f"miss_zero_B{B_miss}"):
            tokens = torch.zeros(
                B_miss, self.K, dtype=torch.int64, device=self.device,
            )
            tokens[:, 0] = bonus_tokens
            logits = torch.zeros(
                B_miss, self.K, self.vocab_size,
                dtype=self.dtype, device=self.device,
            )
        return tokens, logits

