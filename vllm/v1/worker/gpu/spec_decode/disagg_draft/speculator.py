# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify-side proxy for disaggregated (N:M) speculative decoding.

``DisaggSpeculatorProxy`` lives inside the verify server's model runner
and forwards verification outcomes to remote draft servers over ZMQ. It
does not load a draft model locally; the draft servers run as separate
processes on separate GPUs.

The proxy exposes the same ``propose()`` interface as ``EagleSpeculator``
so the model runner can call it uniformly. Under the hood it:

- Uses a ``DraftRouter`` to pick a draft server per request.
- Wraps each connector's async ZMQ calls in a dedicated event loop so
  the synchronous model-runner code path can invoke them.
- Manages per-request seq_id assignment on the draft side; issues
  PREFILL when a request first completes its prompt, FREE_SEQ on
  finish, and SPECULATE every decode step.
- Only TP rank 0 talks to draft servers; other ranks receive the final
  draft tokens via a TP broadcast in the model runner.
"""

from __future__ import annotations

import asyncio
import contextlib
import time as _time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import prometheus_client
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.spec_decode.draft_router import DraftRouter

logger = init_logger(__name__)


class DisaggDraftMetrics:
    """Prometheus metrics for disaggregated speculation (verify side)."""

    def __init__(self) -> None:
        self.draft_tokens_requested = prometheus_client.Counter(
            name="vllm:disagg_draft_tokens_requested_total",
            documentation=("Total draft tokens requested from draft server(s)."),
        )
        self.draft_tokens_accepted = prometheus_client.Counter(
            name="vllm:disagg_draft_tokens_accepted_total",
            documentation=("Total draft tokens accepted after verification."),
        )
        self.draft_round_trip_latency = prometheus_client.Histogram(
            name="vllm:disagg_draft_round_trip_latency_seconds",
            documentation=("Round-trip latency (seconds) for SPECULATE requests."),
            buckets=(
                0.005,
                0.01,
                0.025,
                0.05,
                0.075,
                0.1,
                0.25,
                0.5,
                1.0,
            ),
        )
        self.draft_acceptance_rate = prometheus_client.Gauge(
            name="vllm:disagg_draft_acceptance_rate",
            documentation=(
                "Rolling acceptance rate of draft tokens (accepted / requested)."
            ),
        )
        self._total_requested: int = 0
        self._total_accepted: int = 0

    def record_speculation(
        self,
        tokens_requested: int,
        tokens_accepted: int,
        latency_s: float,
    ) -> None:
        self.draft_tokens_requested.inc(tokens_requested)
        self.draft_tokens_accepted.inc(tokens_accepted)
        self.draft_round_trip_latency.observe(latency_s)

        self._total_requested += tokens_requested
        self._total_accepted += tokens_accepted
        if self._total_requested > 0:
            self.draft_acceptance_rate.set(self._total_accepted / self._total_requested)


class DisaggSpeculatorProxy:
    """Verify-side proxy that relays verification outcomes to remote
    draft servers and returns their pre-computed draft tokens.

    Created by ``init_speculator()`` and stored on the model runner.
    A ``DraftRouter`` is injected via ``set_router()``; construction
    without a router is not useful (all paths require one).
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.num_speculative_steps = self.speculative_config.num_speculative_tokens
        self._is_eagle = self.speculative_config.method == "eagle"
        self.draft_model_config = self.speculative_config.draft_model_config
        self.vocab_size = self.draft_model_config.get_vocab_size()
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.dtype = vllm_config.model_config.dtype

        # Pre-allocated result buffers reused across propose() calls.
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )

        # Path C: pinned CPU staging buffers so the per-round H2Ds in
        # ``_do_propose_dispatch`` don't have to serialize with the
        # target-forward-tail sampler kernel on the default stream.
        # Pageable H2D via ``torch.tensor(list, device=cuda)`` forces
        # such serialization (measured 10 ms wait); pinned + non-blocking
        # ``copy_`` does not.
        self._active_idx_pinned_cpu = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int64,
            pin_memory=True,
        )
        self._seq_ids_pinned_cpu = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int64,
            pin_memory=True,
        )
        self._active_idx_gpu = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int64,
            device=device,
        )
        self._seq_ids_gpu = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int64,
            device=device,
        )
        # Per-server local-indices staging (used in dispatch to pick out
        # a per-server slice of the batch). Same pinned pattern.
        self._srv_idx_pinned_cpu = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int64,
            pin_memory=True,
        )
        self._srv_idx_gpu = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int64,
            device=device,
        )
        self.draft_logits: torch.Tensor | None = None
        if self.speculative_config.rejection_sample_method == "probabilistic":
            self.draft_logits = torch.zeros(
                self.max_num_reqs,
                self.num_speculative_steps,
                self.vocab_size,
                dtype=self.dtype,
                device=device,
            )

        # disagg does not support multimodal inputs for drafting
        self.supports_mm_inputs = False
        self.model = _DisaggDraftModelStub()

        # Injected via set_router(); see init_speculator().
        self.router: DraftRouter | None = None
        self._nm_event_loop: asyncio.AbstractEventLoop | None = None

        # Graceful degradation: reconnection tracking
        self._reconnect_check_interval: int = 10
        self._last_all_unavailable_warn: int = 0
        self._all_unavailable_warn_interval: int = 50

        # Prompt tokens cached for the request lifetime so failover can
        # rebuild draft-side state on a replacement server.
        self._pending_prompt_tokens: dict[str, list[int]] = {}

        # Per-request state for the draft side.
        self._disagg_prefilled_reqs: set[str] = set()
        self._disagg_req_to_seq_id: dict[str, int] = {}
        self._disagg_next_seq_id: int = 0
        self._disagg_free_seq_ids: list[int] = []  # recycled seq_ids

        self._propose_count = 0

        self._metrics = DisaggDraftMetrics()
        self._latency_warn_ms: float = (
            self.speculative_config.disagg_draft_latency_warn_ms
        )

        # Only TP rank 0 talks to the draft server(s). Other ranks
        # short-circuit propose() and rely on a TP broadcast in the
        # model runner to receive the final draft tokens.
        try:
            from vllm.distributed.parallel_state import get_tp_group

            self._tp_rank = get_tp_group().rank_in_group
        except Exception:
            self._tp_rank = 0

        logger.info(
            "DisaggSpeculatorProxy created: K=%d, V=%d, device=%s, tp_rank=%d",
            self.num_speculative_steps,
            self.vocab_size,
            device,
            self._tp_rank,
        )

    # ------------------------------------------------------------------
    # Wiring hooks
    # ------------------------------------------------------------------

    def set_router(self, router: DraftRouter) -> None:
        from vllm.v1.spec_decode.draft_connector import CudaIpcDraftConnector
        from vllm.v1.spec_decode.draft_router import DraftRouter

        assert isinstance(router, DraftRouter)
        self.router = router
        self._nm_event_loop = asyncio.new_event_loop()

        # Perform any deferred IPC handshakes on our newly-created loop
        # so subsequent SPECULATEs can use the fast IPC path. Failure
        # for a given connector logs and leaves it on the ZMQ fallback.
        for i, connector in enumerate(router.connectors):
            if isinstance(connector, CudaIpcDraftConnector):
                try:
                    self._nm_event_loop.run_until_complete(
                        connector.async_establish_ipc()
                    )
                except Exception:
                    logger.exception(
                        "IPC handshake failed for connector %d; "
                        "SPECULATE will use ZMQ fallback",
                        i,
                    )

        logger.info(
            "DisaggSpeculatorProxy: DraftRouter connected with %d server(s).",
            len(router.connectors),
        )

    def cache_new_request_tokens(
        self, req_id: str, prompt_token_ids: list[int]
    ) -> None:
        """Stash a new request's prompt tokens so ``_prefill_new_requests``
        can ship them to a draft server without a second UVA read.

        Tokens remain cached until request completion because server
        failover requires replaying PREFILL on the replacement drafter.
        """
        self._pending_prompt_tokens[req_id] = list(prompt_token_ids)

    @property
    def is_connected(self) -> bool:
        return self.router is not None

    # ------------------------------------------------------------------
    # Graceful degradation
    # ------------------------------------------------------------------

    def _attempt_reconnect_unavailable_servers(self) -> None:
        """Periodically try to reconnect draft servers that have been
        marked unavailable by the router (e.g. after a prior timeout)."""
        if self.router is None:
            return
        for srv_idx, available in enumerate(self.router._available):
            if available:
                continue
            connector = self.router.connectors[srv_idx]
            if not getattr(connector, "connected", False):
                reconnect = getattr(connector, "_reconnect", None)
                if reconnect is not None:
                    with contextlib.suppress(Exception):
                        reconnect()
            if getattr(connector, "connected", False):
                self.router.mark_server_available(srv_idx)
                logger.info(
                    "Draft server %d reconnected successfully.",
                    srv_idx,
                )

    def _handle_server_failure(self, server_index: int) -> None:
        """Invalidate draft-side readiness after routing failover."""
        assert self.router is not None
        affected_req_ids = self.router.handle_server_failure(server_index)
        self._disagg_prefilled_reqs.difference_update(affected_req_ids)

    # ------------------------------------------------------------------
    # Model-runner shim (these methods make the proxy look like a local
    # speculator to the V2 model runner)
    # ------------------------------------------------------------------

    def load_model(self, target_model=None) -> None:
        """No-op: the draft model lives on a separate GPU."""

    def set_attn(self, *args, **kwargs) -> None:
        """No-op: the draft model manages its own attention."""

    def init_cudagraph_manager(self, cudagraph_mode=None) -> None:
        """No-op: CUDA graphs for the draft live on the draft GPU."""

    def capture(self, attn_states=None) -> None:
        """No-op: CUDA graphs for the draft live on the draft GPU."""

    def capture_model(self) -> None:
        """No-op: CUDA graphs for the draft live on the draft GPU."""

    # ------------------------------------------------------------------
    # Propose interfaces for legacy and current GPU model runners
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Split dispatch API (Path C)
    #
    # ``propose_dispatch`` runs steps 1-4 of ``_do_propose`` up through
    # firing SPECULATE. ``propose_await`` waits for the SPECULATE reply
    # and stitches draft tokens into ``self.draft_tokens``. Split so
    # the model runner can overlap the drafter's compute with
    # ``_bookkeeping_sync``'s target-forward-tail wait.
    #
    # The one-shot ``propose`` below still works — it just calls the
    # dispatch + await pair back-to-back. Call sites that don't yet
    # know about the split behave identically to before.
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def propose_dispatch(
        self,
        input_batch,
        num_sampled: torch.Tensor,
        last_sampled: torch.Tensor,
        temperature: torch.Tensor,
        eagle_token_ids: torch.Tensor | None = None,
        eagle_positions: torch.Tensor | None = None,
        eagle_query_lens: torch.Tensor | None = None,
        eagle_hidden_states: torch.Tensor | None = None,
        eagle_next_token_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Fire SPECULATE early (kernel-queued + one CPU doorbell write).

        Returns an opaque context to pass to ``propose_await``. If the
        speculator isn't ready (non-rank-0 TP, no router, warmup, etc.)
        returns None — the caller must handle this by returning zeros.
        """
        if self._tp_rank != 0 or self.router is None:
            return None
        self._propose_count += 1
        if (
            self.router.num_available_servers < len(self.router.connectors)
            and self._propose_count % self._reconnect_check_interval == 0
        ):
            self._attempt_reconnect_unavailable_servers()
        if self.router.num_available_servers == 0:
            return None
        if any(
            rid.startswith("_warmup_") or rid.startswith("_dummy_")
            for rid in input_batch.req_ids
        ):
            return None

        try:
            if self._is_eagle:
                if any(
                    value is None
                    for value in (
                        eagle_token_ids,
                        eagle_positions,
                        eagle_query_lens,
                        eagle_hidden_states,
                        eagle_next_token_ids,
                    )
                ):
                    raise ValueError("Standalone EAGLE target payload is incomplete")
                return self._do_eagle_propose_dispatch(
                    input_batch=input_batch,
                    num_sampled=num_sampled,
                    temperature=temperature,
                    token_ids=eagle_token_ids,
                    positions=eagle_positions,
                    query_lens=eagle_query_lens,
                    hidden_states=eagle_hidden_states,
                    next_token_ids=eagle_next_token_ids,
                )
            return self._do_propose_dispatch(
                input_batch,
                num_sampled,
                last_sampled,
                temperature,
            )
        except Exception:
            logger.exception("propose_dispatch failed; awaiting will zero.")
            return None

    @torch.inference_mode()
    def propose_await(
        self,
        ctx: Any,
        num_reqs: int,
    ) -> torch.Tensor:
        """Wait for the SPECULATE reply, stitch tokens into
        ``self.draft_tokens``, return the ``[num_reqs, K]`` slice.

        If ``ctx`` is None (dispatch skipped or failed), returns zeros.
        """
        K = self.num_speculative_steps
        if ctx is None:
            return torch.zeros(
                num_reqs,
                K,
                dtype=torch.int64,
                device=self.device,
            )
        try:
            return self._do_propose_await(ctx, num_reqs)
        except Exception:
            logger.exception("propose_await failed; returning zeros.")
            return torch.zeros(
                num_reqs,
                K,
                dtype=torch.int64,
                device=self.device,
            )

    @torch.inference_mode()
    def propose(
        self,
        input_batch,
        attn_metadata: dict[str, Any] | torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor] | torch.Tensor | None = None,
        last_hidden_states: torch.Tensor | None = None,
        aux_hidden_states: list[torch.Tensor] | None = None,
        num_sampled: torch.Tensor | None = None,
        num_rejected: torch.Tensor | None = None,
        last_sampled: torch.Tensor | None = None,
        next_prefill_tokens: torch.Tensor | None = None,
        temperature: torch.Tensor | None = None,
        seeds: torch.Tensor | None = None,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        """Request K draft tokens for every request in the batch.

        Supports the canonical ``BaseSpeculator`` call used by
        ``gpu/model_runner.py`` and the legacy positional call
        ``propose(input_batch, num_sampled, last_sampled, temperature)``.

        Returns ``[num_reqs, K]`` int64 tensor. Non-rank-0 TP workers,
        dummy/warmup batches, and batches with no active requests all
        return zeros; callers rely on a TP broadcast to align all ranks.
        """
        if (
            num_sampled is None
            and last_sampled is None
            and temperature is None
            and isinstance(attn_metadata, torch.Tensor)
            and isinstance(slot_mappings, torch.Tensor)
            and isinstance(last_hidden_states, torch.Tensor)
        ):
            num_sampled = attn_metadata
            last_sampled = slot_mappings
            temperature = last_hidden_states

        if num_sampled is None or last_sampled is None or temperature is None:
            raise TypeError(
                "DisaggSpeculatorProxy.propose requires num_sampled, "
                "last_sampled, and temperature."
            )

        num_reqs = input_batch.num_reqs
        K = self.num_speculative_steps
        zeros = torch.zeros(
            num_reqs,
            K,
            dtype=torch.int64,
            device=self.device,
        )

        if dummy_run or is_profile or self._tp_rank != 0 or self.router is None:
            return zeros

        self._propose_count += 1

        # Periodically try to recover unavailable draft servers.
        if (
            self.router.num_available_servers < len(self.router.connectors)
            and self._propose_count % self._reconnect_check_interval == 0
        ):
            self._attempt_reconnect_unavailable_servers()

        if self.router.num_available_servers == 0:
            since = self._propose_count - self._last_all_unavailable_warn
            if since >= self._all_unavailable_warn_interval:
                self._last_all_unavailable_warn = self._propose_count
                logger.warning(
                    "All draft servers unavailable — returning zero "
                    "draft tokens. Will retry reconnection. "
                    "(propose_count=%d)",
                    self._propose_count,
                )
            return zeros

        # Skip warmup/dummy requests — they pollute the draft server's
        # state with fake seq_ids that never get freed.
        if any(
            rid.startswith("_warmup_") or rid.startswith("_dummy_")
            for rid in input_batch.req_ids
        ):
            return zeros

        try:
            if self._is_eagle:
                if (
                    last_hidden_states is None
                    or num_rejected is None
                    or next_prefill_tokens is None
                ):
                    raise TypeError(
                        "MRV2 disaggregated EAGLE requires target hidden states, "
                        "num_rejected, and next_prefill_tokens."
                    )

                scheduled_lens = input_batch.num_scheduled_tokens.tolist()
                rejected_counts = num_rejected[:num_reqs].tolist()
                query_lens_list = [
                    scheduled - int(rejected)
                    for scheduled, rejected in zip(
                        scheduled_lens,
                        rejected_counts,
                    )
                ]
                if any(n <= 0 for n in query_lens_list):
                    return zeros

                gather_indices: list[int] = []
                offset = 0
                for scheduled, valid in zip(scheduled_lens, query_lens_list):
                    gather_indices.extend(range(offset, offset + valid))
                    offset += scheduled

                total_scheduled = input_batch.num_tokens
                if gather_indices == list(range(total_scheduled)):
                    token_ids = input_batch.input_ids[:total_scheduled]
                    positions = input_batch.positions[:total_scheduled]
                    target_hidden_states = last_hidden_states[:total_scheduled]
                else:
                    indices = torch.tensor(
                        gather_indices,
                        dtype=torch.int64,
                        device=self.device,
                    )
                    token_ids = input_batch.input_ids[indices]
                    positions = input_batch.positions[indices]
                    target_hidden_states = last_hidden_states[indices]

                idx_mapping = input_batch.idx_mapping[:num_reqs].long()
                sampled_endpoint = last_sampled[idx_mapping, 0]
                prefill_endpoint = next_prefill_tokens[0, idx_mapping]
                next_token_ids = torch.where(
                    num_sampled[:num_reqs] > 0,
                    sampled_endpoint,
                    prefill_endpoint,
                ).to(torch.int64)
                batch_temperature = temperature[idx_mapping]

                ctx = self._do_eagle_propose_dispatch(
                    input_batch=input_batch,
                    num_sampled=num_sampled,
                    temperature=batch_temperature,
                    token_ids=token_ids,
                    positions=positions,
                    query_lens=torch.tensor(
                        query_lens_list,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    hidden_states=target_hidden_states,
                    next_token_ids=next_token_ids,
                )
                if ctx is None:
                    return zeros
                return self._do_propose_await(ctx, num_reqs)

            return self._do_propose(
                input_batch,
                num_sampled,
                last_sampled,
                temperature,
            )
        except Exception:
            logger.exception("Disagg propose failed; returning zeros.")
            return zeros

    def _do_propose(
        self,
        input_batch,
        num_sampled: torch.Tensor,
        last_sampled: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        from torch.profiler import record_function

        num_reqs = input_batch.num_reqs
        K = self.num_speculative_steps
        req_ids = input_batch.req_ids
        idx_mapping = getattr(input_batch, "idx_mapping", None)
        if idx_mapping is not None:
            idx_mapping = idx_mapping[:num_reqs]
        else:
            idx_mapping = torch.arange(
                num_reqs,
                dtype=torch.int64,
                device=self.device,
            )

        # Step 1: Send FREE_SEQ for requests that have just finished.
        with record_function("propose_step1_free_stale"):
            self._free_stale_requests(req_ids)

        # Step 2: Send PREFILL for new requests that just finished
        # their prompt processing.
        with record_function("propose_step2_check_prefill_needed"):
            new_req_ids = [
                rid
                for i, rid in enumerate(req_ids)
                if rid not in self._disagg_prefilled_reqs
                and int(num_sampled[i].item()) > 0
            ]
        if new_req_ids:
            with record_function("propose_step2b_prefill_new"):
                self._prefill_new_requests(new_req_ids)

        # Step 3: Build the verification outcome tensors over the subset
        # of requests that are prefilled and ready for decode.
        with record_function("propose_step3_build_outcome_tensors"):
            active_req_indices = [
                i
                for i, rid in enumerate(req_ids)
                if rid in self._disagg_prefilled_reqs
                and rid in self._disagg_req_to_seq_id
            ]
            if not active_req_indices:
                return torch.zeros(
                    num_reqs,
                    K,
                    dtype=torch.int64,
                    device=self.device,
                )

            active_req_ids = [req_ids[i] for i in active_req_indices]
            active_idx = torch.tensor(
                active_req_indices,
                dtype=torch.int64,
                device=self.device,
            )
            seq_ids = torch.tensor(
                [self._disagg_req_to_seq_id[rid] for rid in active_req_ids],
                dtype=torch.int64,
                device=self.device,
            )
            k_accepted = (num_sampled[active_idx] - 1).clamp(min=0).to(torch.int64)

            # last_sampled is [num_reqs, max_sampled, 1] or [num_reqs, 1];
            # the bonus token is the last valid sample per request.
            _ls = last_sampled[idx_mapping[active_idx]]
            _ns = num_sampled[active_idx]
            if _ls.dim() == 3:
                last_idx = (_ns - 1).clamp(min=0).long()
                bonus_tokens = _ls[
                    torch.arange(_ls.shape[0], device=_ls.device),
                    last_idx,
                    0,
                ].to(torch.int64)
            elif _ls.dim() == 2:
                last_idx = (_ns - 1).clamp(min=0).long()
                bonus_tokens = _ls[
                    torch.arange(_ls.shape[0], device=_ls.device),
                    last_idx,
                ].to(torch.int64)
            else:
                bonus_tokens = _ls.squeeze(-1).to(torch.int64)

            temps = temperature[idx_mapping[active_idx]].to(torch.float32)

        # Step 4: Dispatch SPECULATE requests to the draft servers.
        import time as _time

        t0 = _time.perf_counter()
        B_active = len(active_req_indices)
        with record_function("propose_step4_dispatch_speculation"):
            draft_toks, draft_logits = self._dispatch_speculation(
                active_req_ids=active_req_ids,
                seq_ids=seq_ids,
                k_accepted=k_accepted,
                bonus_tokens=bonus_tokens,
                temperatures=temps,
                B_active=B_active,
            )
        dt_ms = (_time.perf_counter() - t0) * 1000.0

        with record_function("propose_step5_metrics_sync"):
            tokens_requested = B_active * K
            tokens_accepted = int(k_accepted.sum().item())
        self._metrics.record_speculation(
            tokens_requested=tokens_requested,
            tokens_accepted=tokens_accepted,
            latency_s=dt_ms / 1000.0,
        )
        if dt_ms > self._latency_warn_ms:
            logger.warning(
                "Disagg SPECULATE latency %.2fms exceeds threshold "
                "%.2fms (B_active=%d, K=%d)",
                dt_ms,
                self._latency_warn_ms,
                B_active,
                K,
            )

        # Step 5: Map draft tokens back into the full-batch buffer.
        self.draft_tokens[:num_reqs].zero_()
        if self.draft_logits is not None and draft_logits is not None:
            K_actual = min(draft_logits.shape[1], self.draft_logits.shape[1])
            self.draft_logits[:num_reqs].zero_()
            for j, orig_i in enumerate(active_req_indices):
                self.draft_logits[orig_i, :K_actual] = draft_logits[j, :K_actual]

        target_vocab = self.vllm_config.model_config.get_vocab_size()
        draft_toks = draft_toks.clamp(min=0, max=target_vocab - 1)
        for j, orig_i in enumerate(active_req_indices):
            self.draft_tokens[orig_i] = draft_toks[j]

        return self.draft_tokens[:num_reqs]

    # ------------------------------------------------------------------
    # FREE_SEQ / PREFILL / SPECULATE dispatch helpers
    # ------------------------------------------------------------------

    def _run_async(self, coro):
        """Run an async connector call synchronously on the dedicated loop."""
        assert self._nm_event_loop is not None
        return self._nm_event_loop.run_until_complete(coro)

    def _free_stale_requests(self, req_ids: list[str]) -> None:
        """Send FREE_SEQ per draft server for requests no longer active."""
        assert self.router is not None
        active_rids = set(req_ids)
        tracked_rids = (
            self._disagg_prefilled_reqs
            | set(self._disagg_req_to_seq_id)
            | set(self._pending_prompt_tokens)
            | set(self.router.assignment)
        )
        stale = tracked_rids - active_rids
        if not stale:
            return
        self._disagg_prefilled_reqs -= stale
        stale_by_server: dict[int, list[int]] = defaultdict(list)
        for rid in stale:
            self._pending_prompt_tokens.pop(rid, None)
            sid = self._disagg_req_to_seq_id.pop(rid, None)
            if sid is not None:
                self._disagg_free_seq_ids.append(sid)
                if rid in self.router.assignment:
                    srv_idx = self.router.assignment[rid]
                    stale_by_server[srv_idx].append(sid)
            self.router.release(rid)
        for srv_idx, sids in stale_by_server.items():
            free_ids = torch.tensor(
                sids,
                dtype=torch.int64,
                device=self.device,
            )
            connector = self.router.connectors[srv_idx]
            try:
                self._run_async(connector.send_free_seq(free_ids))
            except Exception as e:
                logger.warning(
                    "FREE_SEQ to draft server %d failed: %s",
                    srv_idx,
                    e,
                )

    def _prefill_new_requests(self, new_req_ids: list[str]) -> None:
        """Assign each new request to a draft server and PREFILL it."""
        assert self.router is not None

        # Allocate seq_ids for new requests (recycled from freed ones
        # when available).
        for rid in new_req_ids:
            if rid in self._disagg_req_to_seq_id:
                continue
            if self._disagg_free_seq_ids:
                sid = self._disagg_free_seq_ids.pop()
            else:
                sid = self._disagg_next_seq_id
                self._disagg_next_seq_id += 1
            self._disagg_req_to_seq_id[rid] = sid

        for rid in new_req_ids:
            prompt_ids = self._pending_prompt_tokens.get(rid)
            if not prompt_ids:
                logger.warning(
                    "Disagg prefill req %s: no cached prompt tokens, skipping.",
                    rid,
                )
                continue

            try:
                connector = self.router.assign(rid)
            except RuntimeError:
                logger.error(
                    "No available draft servers for prefill of req %s",
                    rid,
                )
                continue

            prompt_ids_t = torch.tensor(
                prompt_ids,
                dtype=torch.int64,
                device=self.device,
            )
            try:
                self._run_async(
                    connector.send_prefill(
                        seq_id=self._disagg_req_to_seq_id[rid],
                        prompt_token_ids=prompt_ids_t,
                    )
                )
            except Exception as e:
                logger.error("PREFILL for req %s failed: %s", rid, e)
                if isinstance(e, ConnectionError):
                    server_index = self.router.assignment.get(rid)
                    if server_index is not None:
                        self._handle_server_failure(server_index)
                continue
            self._disagg_prefilled_reqs.add(rid)

    def _do_propose_dispatch(
        self,
        input_batch,
        num_sampled: torch.Tensor,
        last_sampled: torch.Tensor,
        temperature: torch.Tensor,
    ) -> dict | None:
        """Run steps 1-3 of _do_propose and fire SPECULATE early.

        Returns a context dict for propose_await, or None if there's no
        active work to do (empty batch after prefill filtering).
        """
        from torch.profiler import record_function

        from vllm.v1.spec_decode.draft_connector import PendingSpeculation

        num_reqs = input_batch.num_reqs
        req_ids = input_batch.req_ids
        idx_mapping = getattr(input_batch, "idx_mapping", None)
        if idx_mapping is not None:
            idx_mapping = idx_mapping[:num_reqs]
        else:
            idx_mapping = torch.arange(
                num_reqs,
                dtype=torch.int64,
                device=self.device,
            )

        with record_function("propose_step1_free_stale"):
            self._free_stale_requests(req_ids)

        with record_function("propose_step2_check_prefill_needed"):
            new_req_ids = [
                rid
                for i, rid in enumerate(req_ids)
                if rid not in self._disagg_prefilled_reqs
                and int(num_sampled[i].item()) > 0
            ]
        if new_req_ids:
            with record_function("propose_step2b_prefill_new"):
                self._prefill_new_requests(new_req_ids)

        with record_function("propose_step3_build_outcome_tensors"):
            with record_function("propose_step3a_active_req_indices"):
                active_req_indices = [
                    i
                    for i, rid in enumerate(req_ids)
                    if rid in self._disagg_prefilled_reqs
                    and rid in self._disagg_req_to_seq_id
                ]
                if not active_req_indices:
                    return None
                active_req_ids = [req_ids[i] for i in active_req_indices]

            with record_function("propose_step3b_h2d_active_idx"):
                # Fill pinned CPU staging, then async H2D. Non-blocking
                # copies from pinned memory don't serialize with the
                # default stream, so this call returns almost immediately
                # while the target sampler kernel is still running.
                n_active = len(active_req_indices)
                # copy_ from Python list into pinned buffer is a plain
                # memcpy (no CUDA involved).
                self._active_idx_pinned_cpu[:n_active] = torch.as_tensor(
                    active_req_indices,
                    dtype=torch.int64,
                )
                self._seq_ids_pinned_cpu[:n_active] = torch.as_tensor(
                    [self._disagg_req_to_seq_id[rid] for rid in active_req_ids],
                    dtype=torch.int64,
                )
                active_idx = self._active_idx_gpu[:n_active]
                seq_ids = self._seq_ids_gpu[:n_active]
                active_idx.copy_(
                    self._active_idx_pinned_cpu[:n_active],
                    non_blocking=True,
                )
                seq_ids.copy_(
                    self._seq_ids_pinned_cpu[:n_active],
                    non_blocking=True,
                )

            with record_function("propose_step3c_k_accepted"):
                k_accepted = (num_sampled[active_idx] - 1).clamp(min=0).to(torch.int64)

            with record_function("propose_step3d_bonus_tokens"):
                _ls = last_sampled[idx_mapping[active_idx]]
                _ns = num_sampled[active_idx]
                if _ls.dim() == 3:
                    last_idx = (_ns - 1).clamp(min=0).long()
                    bonus_tokens = _ls[
                        torch.arange(_ls.shape[0], device=_ls.device),
                        last_idx,
                        0,
                    ].to(torch.int64)
                elif _ls.dim() == 2:
                    last_idx = (_ns - 1).clamp(min=0).long()
                    bonus_tokens = _ls[
                        torch.arange(_ls.shape[0], device=_ls.device),
                        last_idx,
                    ].to(torch.int64)
                else:
                    bonus_tokens = _ls.squeeze(-1).to(torch.int64)

            with record_function("propose_step3e_temps"):
                temps = temperature[idx_mapping[active_idx]].to(torch.float32)

        # Fire SPECULATE. Fast path: if every connector supports
        # dispatch_speculation returning a real PendingSpeculation (i.e.
        # CudaIpcDraftConnector with IPC ready), do the split. Otherwise
        # we fall back to the base ZMQ path via send_and_recv (in the
        # await step).
        B_active = len(active_req_indices)
        assert self.router is not None
        needs_logits = self.draft_logits is not None

        with record_function("propose_step4_dispatch_early"):
            server_groups: dict[int, list[int]] = defaultdict(list)
            for j, rid in enumerate(active_req_ids):
                if rid in self.router.assignment:
                    srv_idx = self.router.assignment[rid]
                else:
                    self.router.assign(rid)
                    srv_idx = self.router.assignment[rid]
                server_groups[srv_idx].append(j)

            srv_order: list[int] = []
            srv_local_indices: list[list[int]] = []
            handles: list[PendingSpeculation] = []
            single_server_fast_path = len(server_groups) == 1 and next(
                iter(server_groups.values())
            ) == list(range(B_active))
            srv_index_offset = 0
            if not single_server_fast_path:
                grouped_indices = [
                    idx
                    for local_indices in server_groups.values()
                    for idx in local_indices
                ]
                self._srv_idx_pinned_cpu[:B_active] = torch.as_tensor(
                    grouped_indices,
                    dtype=torch.int64,
                )
                self._srv_idx_gpu[:B_active].copy_(
                    self._srv_idx_pinned_cpu[:B_active],
                    non_blocking=True,
                )

            for srv_idx, local_indices in server_groups.items():
                connector = self.router.connectors[srv_idx]
                n = len(local_indices)
                srv_order.append(srv_idx)
                srv_local_indices.append(local_indices)

                if single_server_fast_path:
                    # All requests routed to the same server AND in
                    # order — no need to build a GPU index tensor;
                    # just pass the whole batch slice directly. Avoids
                    # a pageable H2D that would serialize with the
                    # sampler kernel on the default stream.
                    handles.append(
                        connector.dispatch_speculation(
                            batch_size=n,
                            seq_ids=seq_ids[:n],
                            k_accepted=k_accepted[:n],
                            bonus_tokens=bonus_tokens[:n],
                            temperatures=temps[:n],
                            needs_logits=needs_logits,
                        )
                    )
                else:
                    # Each server gets a disjoint slice of the single
                    # asynchronous H2D above. Reusing the same prefix can
                    # overwrite an earlier server's indices before its
                    # dispatch has consumed them.
                    idx_t = self._srv_idx_gpu[srv_index_offset : srv_index_offset + n]
                    srv_index_offset += n
                    handles.append(
                        connector.dispatch_speculation(
                            batch_size=n,
                            seq_ids=seq_ids[idx_t],
                            k_accepted=k_accepted[idx_t],
                            bonus_tokens=bonus_tokens[idx_t],
                            temperatures=temps[idx_t],
                            needs_logits=needs_logits,
                        )
                    )

        return {
            "active_req_indices": active_req_indices,
            "srv_order": srv_order,
            "srv_local_indices": srv_local_indices,
            "handles": handles,
            "B_active": B_active,
            "k_accepted": k_accepted,
            "dispatch_t0": _time.perf_counter(),
        }

    def _do_eagle_propose_dispatch(
        self,
        input_batch,
        num_sampled: torch.Tensor,
        temperature: torch.Tensor,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        query_lens: torch.Tensor,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
    ) -> dict | None:
        """Dispatch one packed target query to a standalone EAGLE server."""
        from vllm.v1.spec_decode.draft_connector import EagleTargetInputs

        assert self.router is not None
        req_ids = input_batch.req_ids
        self._free_stale_requests(req_ids)

        query_lens_list = [int(x) for x in query_lens.tolist()]
        active_req_indices = [i for i, n in enumerate(query_lens_list) if n > 0]
        if not active_req_indices:
            return None
        if active_req_indices != list(range(len(req_ids))):
            raise ValueError(
                "Standalone EAGLE currently requires a positive target query "
                "for every request in the batch"
            )

        for rid in req_ids:
            if rid not in self._disagg_req_to_seq_id:
                if self._disagg_free_seq_ids:
                    sid = self._disagg_free_seq_ids.pop()
                else:
                    sid = self._disagg_next_seq_id
                    self._disagg_next_seq_id += 1
                self._disagg_req_to_seq_id[rid] = sid
            if rid not in self.router.assignment:
                self.router.assign(rid)
            self._disagg_prefilled_reqs.add(rid)

        server_indices = {self.router.assignment[rid] for rid in req_ids}
        if len(server_indices) != 1:
            raise ValueError(
                "A packed standalone EAGLE batch cannot be split across "
                "multiple draft servers"
            )
        srv_idx = next(iter(server_indices))
        connector = self.router.connectors[srv_idx]
        B = len(req_ids)
        seq_ids = torch.tensor(
            [self._disagg_req_to_seq_id[rid] for rid in req_ids],
            dtype=torch.int64,
            device=self.device,
        )
        k_accepted = (num_sampled[:B] - 1).clamp(min=0).to(torch.int64)
        temps = temperature[:B].to(torch.float32)
        handle = connector.dispatch_speculation(
            batch_size=B,
            seq_ids=seq_ids,
            k_accepted=k_accepted,
            bonus_tokens=next_token_ids[:B].to(torch.int64),
            temperatures=temps,
            needs_logits=False,
            eagle_inputs=EagleTargetInputs(
                token_ids=token_ids,
                positions=positions,
                query_lens=query_lens,
                hidden_states=hidden_states,
            ),
        )
        return {
            "active_req_indices": active_req_indices,
            "srv_order": [srv_idx],
            "srv_local_indices": [list(range(B))],
            "handles": [handle],
            "B_active": B,
            "k_accepted": k_accepted,
            "dispatch_t0": _time.perf_counter(),
        }

    def _do_propose_await(self, ctx: dict, num_reqs: int) -> torch.Tensor:
        """Wait for dispatched SPECULATEs, stitch back into
        ``self.draft_tokens``."""
        from torch.profiler import record_function

        K = self.num_speculative_steps
        active_req_indices = ctx["active_req_indices"]
        srv_order = ctx["srv_order"]
        srv_local_indices = ctx["srv_local_indices"]
        handles = ctx["handles"]
        B_active = ctx["B_active"]
        k_accepted = ctx["k_accepted"]
        dispatch_t0 = ctx["dispatch_t0"]

        assert self.router is not None
        router = self.router

        draft_toks_out = torch.zeros(
            B_active,
            K,
            dtype=torch.int64,
            device=self.device,
        )
        draft_logits_out: torch.Tensor | None = None

        async def _gather_awaits():
            coros = [
                router.connectors[srv_idx].await_speculation(h)
                for srv_idx, h in zip(srv_order, handles)
            ]
            return await asyncio.gather(*coros, return_exceptions=True)

        with record_function("propose_await_gather"):
            results = self._run_async(_gather_awaits())

        direct_tokens: torch.Tensor | None = None
        for srv_idx, local_indices, result in zip(
            srv_order,
            srv_local_indices,
            results,
        ):
            if isinstance(result, ConnectionError):
                logger.warning(
                    "Draft server %d failed (%s); marking unavailable.",
                    srv_idx,
                    result,
                )
                self._handle_server_failure(srv_idx)
                continue
            if isinstance(result, BaseException):
                logger.warning(
                    "Draft server %d error: %s; affected requests get zero drafts.",
                    srv_idx,
                    result,
                )
                continue

            _, srv_draft_toks, srv_draft_logits = result
            if (
                len(results) == 1
                and local_indices == list(range(B_active))
                and srv_draft_toks.shape[0] >= B_active
            ):
                direct_tokens = srv_draft_toks[:B_active]
            else:
                for local_j, global_j in enumerate(local_indices):
                    if local_j < srv_draft_toks.shape[0]:
                        draft_toks_out[global_j] = srv_draft_toks[local_j]
            if srv_draft_logits is not None:
                if draft_logits_out is None:
                    draft_logits_out = torch.zeros(
                        B_active,
                        K,
                        self.vocab_size,
                        dtype=self.dtype,
                        device=self.device,
                    )
                for local_j, global_j in enumerate(local_indices):
                    if local_j < srv_draft_logits.shape[0]:
                        K_actual = min(
                            srv_draft_logits.shape[1],
                            draft_logits_out.shape[1],
                        )
                        draft_logits_out[global_j, :K_actual] = srv_draft_logits[
                            local_j, :K_actual
                        ]

        dt_ms = (_time.perf_counter() - dispatch_t0) * 1000.0
        tokens_requested = B_active * K
        tokens_accepted = int(k_accepted.sum().item())
        self._metrics.record_speculation(
            tokens_requested=tokens_requested,
            tokens_accepted=tokens_accepted,
            latency_s=dt_ms / 1000.0,
        )
        if dt_ms > self._latency_warn_ms:
            logger.warning(
                "Disagg SPECULATE latency %.2fms exceeds threshold "
                "%.2fms (B_active=%d, K=%d)",
                dt_ms,
                self._latency_warn_ms,
                B_active,
                K,
            )

        # Stitch into self.draft_tokens
        self.draft_tokens[:num_reqs].zero_()
        if self.draft_logits is not None and draft_logits_out is not None:
            K_actual = min(
                draft_logits_out.shape[1],
                self.draft_logits.shape[1],
            )
            self.draft_logits[:num_reqs].zero_()
            for j, orig_i in enumerate(active_req_indices):
                self.draft_logits[orig_i, :K_actual] = draft_logits_out[j, :K_actual]

        target_vocab = self.vllm_config.model_config.get_vocab_size()
        if direct_tokens is not None:
            draft_toks_out = direct_tokens
        draft_toks_out = draft_toks_out.clamp(min=0, max=target_vocab - 1)
        if active_req_indices == list(range(num_reqs)):
            self.draft_tokens[:num_reqs].copy_(draft_toks_out[:num_reqs])
        else:
            for j, orig_i in enumerate(active_req_indices):
                self.draft_tokens[orig_i] = draft_toks_out[j]

        return self.draft_tokens[:num_reqs]

    def _dispatch_speculation(
        self,
        active_req_ids: list[str],
        seq_ids: torch.Tensor,
        k_accepted: torch.Tensor,
        bonus_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        B_active: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Group requests by draft server, send per-server SPECULATEs
        concurrently, stitch results back to active-batch order."""
        from torch.profiler import record_function

        assert self.router is not None
        K = self.num_speculative_steps

        with record_function("disp_group_by_server"):
            server_groups: dict[int, list[int]] = defaultdict(list)
            for j, rid in enumerate(active_req_ids):
                if rid in self.router.assignment:
                    srv_idx = self.router.assignment[rid]
                else:
                    self.router.assign(rid)
                    srv_idx = self.router.assignment[rid]
                server_groups[srv_idx].append(j)

            draft_toks_out = torch.zeros(
                B_active,
                K,
                dtype=torch.int64,
                device=self.device,
            )
            draft_logits_out: torch.Tensor | None = None
            needs_logits = self.draft_logits is not None

            srv_order: list[int] = []
            srv_local_indices: list[list[int]] = []
            coros: list[Any] = []
            for srv_idx, local_indices in server_groups.items():
                connector = self.router.connectors[srv_idx]
                n = len(local_indices)
                idx_t = torch.tensor(
                    local_indices,
                    dtype=torch.int64,
                    device=self.device,
                )
                srv_order.append(srv_idx)
                srv_local_indices.append(local_indices)
                coros.append(
                    connector.send_and_recv_speculation(
                        batch_size=n,
                        seq_ids=seq_ids[idx_t],
                        k_accepted=k_accepted[idx_t],
                        bonus_tokens=bonus_tokens[idx_t],
                        temperatures=temperatures[idx_t],
                        needs_logits=needs_logits,
                    )
                )

        if not coros:
            return draft_toks_out, draft_logits_out

        async def _gather():
            return await asyncio.gather(*coros, return_exceptions=True)

        with record_function("disp_run_async_gather"):
            results = self._run_async(_gather())
        for srv_idx, local_indices, result in zip(
            srv_order, srv_local_indices, results
        ):
            if isinstance(result, ConnectionError):
                logger.warning(
                    "Draft server %d failed (%s); marking unavailable.",
                    srv_idx,
                    result,
                )
                self._handle_server_failure(srv_idx)
                continue
            if isinstance(result, BaseException):
                logger.warning(
                    "Draft server %d error: %s; affected requests get zero drafts.",
                    srv_idx,
                    result,
                )
                continue

            _, srv_draft_toks, srv_draft_logits = result
            for local_j, global_j in enumerate(local_indices):
                if local_j < srv_draft_toks.shape[0]:
                    draft_toks_out[global_j] = srv_draft_toks[local_j]

            if srv_draft_logits is not None:
                if draft_logits_out is None:
                    draft_logits_out = torch.zeros(
                        B_active,
                        K,
                        self.vocab_size,
                        dtype=self.dtype,
                        device=self.device,
                    )
                for local_j, global_j in enumerate(local_indices):
                    if local_j < srv_draft_logits.shape[0]:
                        K_actual = min(
                            srv_draft_logits.shape[1],
                            draft_logits_out.shape[1],
                        )
                        draft_logits_out[global_j, :K_actual] = srv_draft_logits[
                            local_j, :K_actual
                        ]

        return draft_toks_out, draft_logits_out


class _DisaggDraftModelStub(torch.nn.Module):
    """Stub exposed via ``speculator.model`` so the model runner's
    generic handling (e.g. eplb registration) doesn't crash. The real
    draft model lives on a separate process on a separate GPU."""

    def __init__(self):
        super().__init__()
