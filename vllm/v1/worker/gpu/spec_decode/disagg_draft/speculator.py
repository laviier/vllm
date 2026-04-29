# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Disaggregated Draft Speculator — Target-side speculator for disagg draft spec decoding (V2 Model Runner).

This class runs on the target GPU (inside GPUModelRunner V2) and acts as a
thin proxy to the disaggregated draft worker running on a separate GPU.
It implements the same interface as EagleSpeculator so the model runner
can call it identically.

Unlike Eagle (which runs the draft model on the same GPU), disagg_draft:
- Does NOT load a draft model on the target GPU
- Does NOT manage attention or KV cache for drafting
- Does NOT use CUDA graphs for drafting
- Instead, sends verification outcomes to the draft worker via NCCL
  and receives pre-computed draft tokens back
"""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn

import prometheus_client

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

_DISAGG_DEBUG = os.environ.get("DISAGG_EAGLE_DEBUG", "0") == "1"


class DisaggDraftMetrics:
    """Prometheus metrics for disaggregated draft speculation on the
    verify server side.

    Tracks draft tokens requested/accepted, round-trip latency, and
    a rolling acceptance rate gauge.
    """

    def __init__(self) -> None:
        self.draft_tokens_requested = prometheus_client.Counter(
            name="vllm:disagg_draft_tokens_requested_total",
            documentation=(
                "Total draft tokens requested from draft server(s)."
            ),
        )
        self.draft_tokens_accepted = prometheus_client.Counter(
            name="vllm:disagg_draft_tokens_accepted_total",
            documentation=(
                "Total draft tokens accepted after verification."
            ),
        )
        self.draft_round_trip_latency = prometheus_client.Histogram(
            name="vllm:disagg_draft_round_trip_latency_seconds",
            documentation=(
                "Round-trip latency (seconds) for draft speculation "
                "requests."
            ),
            buckets=(
                0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0,
            ),
        )
        self.draft_acceptance_rate = prometheus_client.Gauge(
            name="vllm:disagg_draft_acceptance_rate",
            documentation=(
                "Rolling acceptance rate of draft tokens "
                "(accepted / requested)."
            ),
        )
        # Internal accumulators for computing rolling acceptance rate.
        self._total_requested: int = 0
        self._total_accepted: int = 0

    def record_speculation(
        self,
        tokens_requested: int,
        tokens_accepted: int,
        latency_s: float,
    ) -> None:
        """Record metrics for a single speculation round."""
        self.draft_tokens_requested.inc(tokens_requested)
        self.draft_tokens_accepted.inc(tokens_accepted)
        self.draft_round_trip_latency.observe(latency_s)

        self._total_requested += tokens_requested
        self._total_accepted += tokens_accepted
        if self._total_requested > 0:
            self.draft_acceptance_rate.set(
                self._total_accepted / self._total_requested
            )


class DisaggSpeculatorProxy:
    """Unified target-side proxy for all disaggregated speculation methods.

    Created by init_speculator() and stored in GPUModelRunner.speculator.
    On the first propose() call, lazily establishes the NCCL PG to the
    draft worker.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.num_speculative_steps = (
            self.speculative_config.num_speculative_tokens
        )
        self.draft_model_config = self.speculative_config.draft_model_config
        self.vocab_size = self.draft_model_config.get_vocab_size()
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.dtype = vllm_config.model_config.dtype

        # Pre-allocate draft token buffer
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )

        # Draft logits for rejection sampling (None = strict mode)
        self.draft_logits: torch.Tensor | None = None
        if self.speculative_config.rejection_sample_method == "probabilistic":
            self.draft_logits = torch.zeros(
                self.max_num_reqs,
                self.num_speculative_steps,
                self.vocab_size,
                dtype=self.dtype,
                device=device,
            )

        # Auto-detect whether the method requires hidden state transfer
        self.needs_hidden_states = (
            self.speculative_config.disagg_needs_hidden_states
        )
        self.hidden_size = vllm_config.model_config.get_hidden_size()

        # For EAGLE3, the transfer hidden size is num_aux_layers * target_hidden_size
        # (concatenated aux hidden states). For EAGLE/MTP, it's just hidden_size.
        if self.speculative_config.method == "eagle3":
            # Determine number of aux layers from config or default (3)
            from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
                get_eagle3_aux_layers_from_config,
            )
            aux_layers = get_eagle3_aux_layers_from_config(
                self.speculative_config)
            num_aux = len(aux_layers) if aux_layers else 3
            # Use target_hidden_size from draft config if available,
            # otherwise fall back to base model hidden_size.
            draft_hf = self.speculative_config.draft_model_config.hf_config
            target_hs = getattr(draft_hf, 'target_hidden_size',
                                self.hidden_size)
            self.transfer_hidden_size = num_aux * target_hs
        else:
            self.transfer_hidden_size = self.hidden_size

        # disagg does not support multimodal inputs for drafting
        self.supports_mm_inputs = False

        # DisaggDraftTargetInterface — lazily created on first propose()
        self._target_interface = None
        self._nccl_connect_attempted = False

        # N:M DraftRouter — set via set_router() when uses_nm_disagg
        self.router: "DraftRouter | None" = None
        # Dedicated event loop for bridging async DraftConnector calls
        # from synchronous _do_propose / _prefill_new_requests.
        self._nm_event_loop: asyncio.AbstractEventLoop | None = None

        # Graceful degradation: reconnection tracking
        self._reconnect_check_interval: int = 10  # check every N calls
        self._last_all_unavailable_warn: int = 0
        self._all_unavailable_warn_interval: int = 50  # warn every N calls

        # Reference to model runner's request states (set via set_req_states)
        self._req_states = None

        # Cache of prompt tokens for new requests, keyed by req_id.
        # Populated by cache_new_request_tokens(), consumed by
        # _prefill_new_requests().
        self._pending_prompt_tokens: dict[str, list[int]] = {}

        # Track per-sequence state for the draft worker
        self._disagg_prefilled_reqs: set[str] = set()
        self._disagg_req_to_seq_id: dict[str, int] = {}
        self._disagg_next_seq_id: int = 0
        self._disagg_free_seq_ids: list[int] = []  # recycled seq_ids

        self._propose_count = 0

        # Metrics and latency warning threshold
        self._metrics = DisaggDraftMetrics()
        self._latency_warn_ms: float = (
            self.speculative_config.disagg_draft_latency_warn_ms
        )

        # Determine TP rank — only rank 0 communicates with draft worker.
        try:
            from vllm.distributed.parallel_state import get_tp_group
            self._tp_rank = get_tp_group().rank_in_group
        except Exception:
            self._tp_rank = 0

        logger.info(
            "DisaggSpeculatorProxy created: K=%d, V=%d, device=%s, "
            "tp_rank=%d, needs_hidden_states=%s, hidden_size=%d, "
            "transfer_hidden_size=%d",
            self.num_speculative_steps,
            self.vocab_size,
            device,
            self._tp_rank,
            self.needs_hidden_states,
            self.hidden_size,
            self.transfer_hidden_size,
        )

        if (self.needs_hidden_states
                and vllm_config.cache_config.enable_prefix_caching):
            logger.warning(
                "Disagg EAGLE with prefix caching enabled: acceptance "
                "rate will be degraded for prefix-cached requests because "
                "the EAGLE head cannot access hidden states for cached "
                "prefix tokens. Consider --no-enable-prefix-caching for "
                "best speculation accuracy."
            )

    def set_target_interface(self, interface) -> None:
        """Inject the DisaggDraftTargetInterface after NCCL PG setup."""
        self._target_interface = interface
        logger.info("DisaggSpeculatorProxy: target interface connected.")

    def set_router(self, router: "DraftRouter") -> None:
        """Inject the DraftRouter for N:M disaggregated speculation.

        When a router is set, the proxy routes verification outcomes
        and prefill requests through ``DraftRouter`` → ``DraftConnector``
        instead of the 1:1 NCCL ``DisaggDraftTargetInterface``.
        """
        from vllm.v1.spec_decode.draft_router import DraftRouter
        assert isinstance(router, DraftRouter)
        self.router = router
        # Create a dedicated event loop for async connector calls.
        self._nm_event_loop = asyncio.new_event_loop()
        logger.info(
            "DisaggSpeculatorProxy: DraftRouter connected with %d server(s).",
            len(router.connectors),
        )

    def set_req_states(self, req_states) -> None:
        """Store reference to model runner's RequestState for token access."""
        self._req_states = req_states

    def cache_new_request_tokens(
        self, req_id: str, prompt_token_ids: list[int]
    ) -> None:
        """Cache prompt tokens for a new request.

        Called from the model runner's add_requests() so the speculator
        has the actual prompt tokens available when propose() runs later.
        """
        self._pending_prompt_tokens[req_id] = list(prompt_token_ids)

    def _extract_batch_hidden_states(
        self,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        input_batch,
        num_sampled: torch.Tensor,
        new_req_ids: set | None = None,
        active_indices: list[int] | None = None,
    ) -> torch.Tensor:
        """Extract per-request hidden states from model output.

        Args:
            last_hidden_states: [num_tokens, hidden_size]
            aux_hidden_states: list of [num_tokens, hidden_size] for EAGLE3
            input_batch: input batch with query_start_loc
            num_sampled: [num_reqs] tokens accepted per request
            new_req_ids: set of req_ids that were just prefilled this round
            active_indices: indices into input_batch for active requests.
                If None, uses all requests.
        """
        if active_indices is None:
            indices = list(range(input_batch.num_reqs))
        else:
            indices = active_indices
        B = len(indices)

        last_token_indices = torch.zeros(B, dtype=torch.long, device=last_hidden_states.device)
        for j, orig_i in enumerate(indices):
            rid = input_batch.req_ids[orig_i]
            ns = int(num_sampled[orig_i].item())
            if new_req_ids and rid in new_req_ids:
                # Prefill: use last query token
                last_token_indices[j] = (
                    input_batch.query_start_loc[orig_i + 1] - 1
                )
            else:
                # Decode: use last accepted position
                last_token_indices[j] = (
                    input_batch.query_start_loc[orig_i] + ns - 1
                )

        if aux_hidden_states is not None and len(aux_hidden_states) > 0:
            combined = torch.cat(list(aux_hidden_states), dim=-1)
            hs = combined[last_token_indices]
            if _DISAGG_DEBUG:
                logger.info(
                    "[DISAGG_DIAG][CP1] path=aux_hidden_states "
                    "last_token_indices=%s source_shape=%s",
                    last_token_indices.tolist(), list(combined.shape))
                for j in range(hs.shape[0]):
                    logger.info(
                        "[DISAGG_DIAG][CP1] req=%d norm=%.6f "
                        "dtype=%s first3=%s",
                        j, hs[j].float().norm().item(),
                        hs.dtype, hs[j, :3].tolist())
            return hs

        hs = last_hidden_states[last_token_indices]
        if _DISAGG_DEBUG:
            logger.info(
                "[DISAGG_DIAG][CP1] path=last_hidden_states "
                "last_token_indices=%s source_shape=%s",
                last_token_indices.tolist(),
                list(last_hidden_states.shape))
            for j in range(hs.shape[0]):
                logger.info(
                    "[DISAGG_DIAG][CP1] req=%d norm=%.6f "
                    "dtype=%s first3=%s",
                    j, hs[j].float().norm().item(),
                    hs.dtype, hs[j, :3].tolist())
        return hs

    @property
    def is_connected(self) -> bool:
        return self._target_interface is not None or self.router is not None

    # ------------------------------------------------------------------
    # Graceful degradation helpers
    # ------------------------------------------------------------------

    def _attempt_reconnect_unavailable_servers(self) -> None:
        """Periodically check unavailable draft servers and reconnect.

        Called every ``_reconnect_check_interval`` propose() calls when
        the router has at least one unavailable server.  For each
        unavailable server whose connector reports ``connected == False``,
        we call ``_reconnect()`` and, on success, mark the server
        available again in the router.
        """
        if self.router is None:
            return

        for srv_idx, available in enumerate(self.router._available):
            if available:
                continue
            connector = self.router.connectors[srv_idx]
            # Use the connector's own connected property to check state
            if not getattr(connector, 'connected', False):
                try:
                    connector._reconnect()
                except Exception:
                    pass
            if getattr(connector, 'connected', False):
                self.router.mark_server_available(srv_idx)
                logger.info(
                    "Draft server %d reconnected successfully.",
                    srv_idx,
                )

    def _lazy_connect(self) -> bool:
        """Try to lazily establish the NCCL PG to the draft worker.
        Only TP rank 0 connects; other ranks skip."""
        if self._nccl_connect_attempted:
            return self.is_connected
        self._nccl_connect_attempted = True

        # Only TP rank 0 communicates with the draft worker.
        if self._tp_rank != 0:
            logger.info("Disagg draft: TP rank %d — skipping NCCL connect.", self._tp_rank)
            return False

        nccl_init_method = self.speculative_config.disagg_nccl_init_method
        if not nccl_init_method:
            logger.warning(
                "Disagg draft: disagg_nccl_init_method not set. "
                "Draft worker connection not available."
            )
            return False

        try:
            from datetime import timedelta
            from urllib.parse import urlparse

            import torch.distributed as dist
            from vllm.v1.worker.gpu.spec_decode.disagg_draft.communication import (
                DisaggDraftCommunicator,
            )
            from vllm.v1.worker.gpu.spec_decode.disagg_draft.draft_worker import (
                DisaggDraftTargetInterface,
            )

            logger.info(
                "Disagg draft: Connecting to draft worker via %s", nccl_init_method
            )

            parsed = urlparse(nccl_init_method)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 29500

            store = dist.TCPStore(
                host_name=host,
                port=port,
                world_size=2,
                is_master=False,
                timeout=timedelta(seconds=120),
            )
            disagg_pg = dist.ProcessGroupNCCL(
                store, rank=0, size=2,
                timeout=timedelta(hours=24),
            )

            spec_config = self.speculative_config
            communicator = DisaggDraftCommunicator(
                process_group=disagg_pg,
                peer_rank=1,
                num_speculative_tokens=spec_config.num_speculative_tokens,
                max_batch_size=self.max_num_reqs,
                vocab_size=self.vocab_size,
                device=self.device,
                dtype=self.dtype,
                needs_hidden_states=self.needs_hidden_states,
                hidden_size=self.transfer_hidden_size,
            )

            self._target_interface = DisaggDraftTargetInterface(
                communicator=communicator,
                num_speculative_tokens=spec_config.num_speculative_tokens,
                vocab_size=self.vocab_size,
                device=self.device,
                dtype=self.dtype,
            )

            logger.info("Disagg draft: Target interface connected to draft worker.")
            return True

        except Exception:
            logger.exception("Disagg draft: Failed to connect to draft worker.")
            return False

    # ------------------------------------------------------------------
    # Interface methods matching EagleSpeculator (V2 Model Runner)
    # ------------------------------------------------------------------

    def load_model(self, target_model: nn.Module) -> None:
        """No-op: disagg draft draft model lives on the draft GPU."""
        self.model = _DisaggDraftModelStub()
        logger.info("DisaggSpeculatorProxy.load_model: no-op (draft on separate GPU)")

    def set_attn(self, model_state, kv_cache_config,
                 block_tables) -> None:
        """No-op: disagg draft draft model manages its own attention."""
        pass

    def capture_model(self) -> None:
        """No-op: CUDA graphs for draft model are on draft GPU."""
        pass

    @torch.inference_mode()
    def propose(
        self,
        input_batch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Generate draft tokens by communicating with the draft worker.
        Only TP rank 0 does NCCL communication; other ranks return zeros."""
        num_reqs = input_batch.num_reqs
        K = self.num_speculative_steps

        if dummy_run or self._tp_rank != 0:
            return torch.zeros(
                num_reqs, K,
                dtype=torch.int64, device=self.device,
            )

        if not self.is_connected:
            # Only attempt lazy NCCL connect for 1:1 mode (no router).
            if self.router is None:
                self._lazy_connect()

        if not self.is_connected:
            return torch.zeros(
                num_reqs, K,
                dtype=torch.int64, device=self.device,
            )

        self._propose_count += 1

        # --- Graceful degradation: reconnect unavailable servers ---
        if (self.router is not None
                and self.router.num_available_servers < len(
                    self.router.connectors)
                and self._propose_count
                    % self._reconnect_check_interval == 0):
            self._attempt_reconnect_unavailable_servers()

        # --- Graceful degradation: all servers unavailable ---
        if (self.router is not None
                and self.router.num_available_servers == 0):
            since_last = (
                self._propose_count - self._last_all_unavailable_warn
            )
            if since_last >= self._all_unavailable_warn_interval:
                self._last_all_unavailable_warn = self._propose_count
                logger.warning(
                    "All draft servers unavailable — returning zero "
                    "draft tokens (no speculation). Will retry "
                    "reconnection periodically. (propose_count=%d)",
                    self._propose_count,
                )
            return torch.zeros(
                num_reqs, K,
                dtype=torch.int64, device=self.device,
            )

        # Skip warmup/dummy requests — they pollute the draft worker's
        # state with fake seq_ids and KV cache entries.
        if any(rid.startswith('_warmup_') or rid.startswith('_dummy_')
               for rid in input_batch.req_ids):
            return torch.zeros(
                num_reqs, K,
                dtype=torch.int64, device=self.device,
            )

        try:
            return self._do_propose(
                input_batch, num_sampled, num_rejected, last_sampled,
                temperature, last_hidden_states, aux_hidden_states,
            )
        except Exception as e:
            logger.error("Disagg draft propose FAILED: %s", e, exc_info=True)
            return torch.zeros(
                num_reqs, K,
                dtype=torch.int64, device=self.device,
            )

    def _do_propose(
        self,
        input_batch,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        temperature: torch.Tensor,
        last_hidden_states: torch.Tensor | None = None,
        aux_hidden_states: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Core propose logic: manage sequences and communicate with draft."""
        num_reqs = input_batch.num_reqs
        req_ids = input_batch.req_ids
        idx_mapping = getattr(input_batch, 'idx_mapping', None)
        if idx_mapping is not None:
            idx_mapping = idx_mapping[:num_reqs]
        else:
            # Fallback: identity mapping when called from gpu_model_runner
            # inline code (InputBatch doesn't have idx_mapping).
            idx_mapping = torch.arange(
                num_reqs, dtype=torch.int64, device=self.device
            )

        # --- Step 1: Clean up finished requests ---
        active_rids = set(req_ids)
        stale = self._disagg_prefilled_reqs - active_rids
        if stale:
            self._disagg_prefilled_reqs -= stale
            if self.router is not None:
                # N:M mode: group stale requests by their assigned
                # connector and send FREE_SEQ per-connector.
                stale_by_server: dict[int, list[int]] = defaultdict(list)
                for rid in stale:
                    sid = self._disagg_req_to_seq_id.pop(rid, None)
                    if sid is not None:
                        self._disagg_free_seq_ids.append(sid)
                        if rid in self.router.assignment:
                            srv_idx = self.router.assignment[rid]
                            stale_by_server[srv_idx].append(sid)
                        else:
                            # Request not assigned — nothing to free
                            pass
                    self.router.release(rid)
                for srv_idx, sids in stale_by_server.items():
                    free_ids = torch.tensor(
                        sids, dtype=torch.int64, device=self.device,
                    )
                    connector = self.router.connectors[srv_idx]
                    try:
                        self._run_async(
                            connector.send_free_seq(free_ids))
                    except Exception as e:
                        logger.warning(
                            "N:M free_seq to server %d failed: %s",
                            srv_idx, e,
                        )
            else:
                # 1:1 mode: existing path
                stale_seq_ids = []
                for rid in stale:
                    sid = self._disagg_req_to_seq_id.pop(rid, None)
                    if sid is not None:
                        stale_seq_ids.append(sid)
                        self._disagg_free_seq_ids.append(sid)
                if stale_seq_ids:
                    free_ids = torch.tensor(
                        stale_seq_ids, dtype=torch.int64,
                        device=self.device,
                    )
                    self._target_interface.request_free_seq(free_ids)

        # --- Step 2: Prefill new requests on the draft worker ---
        # Only prefill requests that have completed their prompt
        # processing (num_sampled > 0). With chunked prefill, a
        # request may appear in the batch before all prompt tokens
        # are processed. We must wait until the full prompt is done.
        new_req_ids = []
        for i, rid in enumerate(req_ids):
            if rid not in self._disagg_prefilled_reqs:
                # Check if this request has finished prefill
                # (num_sampled > 0 means at least one token was sampled)
                if int(num_sampled[i].item()) > 0:
                    new_req_ids.append(rid)
        if new_req_ids:
            self._prefill_new_requests(
                input_batch, new_req_ids,
                last_hidden_states=last_hidden_states,
                aux_hidden_states=aux_hidden_states,
            )

        # --- Step 3: Build verification outcome ---
        # Only include requests that have been prefilled on the draft
        # worker. Requests still in chunked prefill (num_sampled=0)
        # don't have a seq_id yet.
        active_req_indices = [
            i for i, rid in enumerate(req_ids)
            if rid in self._disagg_req_to_seq_id
        ]
        if not active_req_indices:
            return torch.zeros(
                num_reqs, K,
                dtype=torch.int64, device=self.device,
            )

        active_req_ids = [req_ids[i] for i in active_req_indices]
        seq_ids = torch.tensor(
            [self._disagg_req_to_seq_id[rid] for rid in active_req_ids],
            dtype=torch.int64, device=self.device,
        )
        active_idx = torch.tensor(
            active_req_indices, dtype=torch.int64, device=self.device)

        k_accepted = (num_sampled[active_idx] - 1).clamp(min=0).to(torch.int64)
        # last_sampled has shape [num_reqs, max_sampled, 1] or [num_reqs, 1].
        # The bonus token is the LAST valid sampled token per request.
        _ls = last_sampled[idx_mapping[active_idx]]  # [B_active, T, 1] or [B_active, 1]
        _ns = num_sampled[active_idx]  # [B_active]
        if _ls.dim() == 3:
            # [B_active, T, 1] → pick last valid token per row → [B_active]
            last_idx = (_ns - 1).clamp(min=0).long()  # [B_active]
            bonus_tokens = _ls[
                torch.arange(_ls.shape[0], device=_ls.device),
                last_idx,
                0,
            ].to(torch.int64)
        elif _ls.dim() == 2:
            # [B_active, T] → pick last valid token per row → [B_active]
            last_idx = (_ns - 1).clamp(min=0).long()
            bonus_tokens = _ls[
                torch.arange(_ls.shape[0], device=_ls.device),
                last_idx,
            ].to(torch.int64)
        else:
            bonus_tokens = _ls.squeeze(-1).to(torch.int64)

        # --- Step 3b: Build extend data for glue decode ---
        # The extend data contains the target's hidden states for the
        # tokens accepted in the CURRENT verification. These are sent
        # to the draft worker so it can run glue decode to fill KV
        # cache gaps.
        #
        # EAGLE conditioning shift: token at position p gets conditioning
        # from hidden state at position p-1. So for n_ext accepted draft
        # tokens, we need:
        #   - Token IDs: the accepted draft tokens (positions 1..n_ext)
        #   - Hidden states: positions 0..n_ext-1 (shifted by 1)
        #     Position 0 = recovery token's hs → conditions first draft
        #     Position 1 = first draft's hs → conditions second draft
        #     etc.
        K = self.num_speculative_steps
        B_active = len(active_req_indices)
        extend_counts = None
        extend_hidden_states = None
        extend_token_ids = None
        if self.needs_hidden_states and aux_hidden_states is not None:
            extend_counts = torch.zeros(
                B_active, dtype=torch.int64, device=self.device)
            extend_hidden_states = torch.zeros(
                B_active, K, self.transfer_hidden_size,
                dtype=self.dtype, device=self.device)
            extend_token_ids = torch.zeros(
                B_active, K, dtype=torch.int64, device=self.device)
            combined = torch.cat(list(aux_hidden_states), dim=-1)
            for j, orig_i in enumerate(active_req_indices):
                k_acc = int(k_accepted[j].item())
                n_ext = min(k_acc, K)
                if n_ext > 0:
                    start = int(
                        input_batch.query_start_loc[orig_i].item())
                    extend_hidden_states[j, :n_ext] = (
                        combined[start:start + n_ext])
                    extend_token_ids[j, :n_ext] = (
                        input_batch.input_ids[
                            start + 1:start + 1 + n_ext
                        ].to(torch.int64))
                    extend_counts[j] = n_ext

        # --- Step 4: Request speculation from draft worker ---
        temps = temperature[idx_mapping[active_idx]].to(torch.float32)

        # Extract per-request hidden states for EAGLE/EAGLE3/MTP methods
        hs = None
        if self.needs_hidden_states and last_hidden_states is not None:
            hs = self._extract_batch_hidden_states(
                last_hidden_states, aux_hidden_states, input_batch,
                num_sampled,
                new_req_ids=set(new_req_ids) if new_req_ids else None,
                active_indices=active_req_indices,
            )

        import time as _time
        _t0 = _time.perf_counter()

        if self.router is not None:
            # ---- N:M mode: route through DraftRouter → DraftConnector ----
            draft_toks, draft_logits = self._do_propose_nm(
                active_req_ids=active_req_ids,
                active_req_indices=active_req_indices,
                seq_ids=seq_ids,
                k_accepted=k_accepted,
                bonus_tokens=bonus_tokens,
                temperatures=temps,
                hidden_states=hs,
                extend_counts=extend_counts,
                extend_hidden_states=extend_hidden_states,
                extend_token_ids=extend_token_ids,
                B_active=B_active,
            )
        else:
            # ---- 1:1 mode: existing NCCL path ----
            _, draft_toks, draft_logits = \
                self._target_interface.request_speculation(
                    seq_ids=seq_ids,
                    k_accepted=k_accepted,
                    bonus_tokens=bonus_tokens,
                    batch_size=B_active,
                    temperatures=temps,
                    hidden_states=hs,
                    extend_counts=extend_counts,
                    extend_hidden_states=extend_hidden_states,
                    extend_token_ids=extend_token_ids,
                )

        _dt = (_time.perf_counter() - _t0) * 1000
        if not hasattr(self, '_spec_times'):
            self._spec_times = []
        self._spec_times.append(_dt)
        if len(self._spec_times) % 200 == 0:
            avg = sum(self._spec_times[-200:]) / 200
            logger.info(
                "Disagg SPECULATE latency: avg=%.2fms over last 200 calls "
                "(total %d calls)",
                avg, len(self._spec_times),
            )

        # --- Record metrics ---
        _dt_s = _dt / 1000.0  # convert ms to seconds for Histogram
        tokens_requested = B_active * K
        # tokens_accepted = sum of k_accepted across active requests
        tokens_accepted = int(k_accepted.sum().item())
        self._metrics.record_speculation(
            tokens_requested=tokens_requested,
            tokens_accepted=tokens_accepted,
            latency_s=_dt_s,
        )
        if _dt > self._latency_warn_ms:
            logger.warning(
                "Disagg draft round-trip latency %.2fms exceeds threshold "
                "%.2fms (B_active=%d, K=%d)",
                _dt, self._latency_warn_ms, B_active, K,
            )

        # Map draft tokens back to the full batch.
        # Requests not in active_req_indices get zeros (no draft tokens).
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
    # N:M routing helpers
    # ------------------------------------------------------------------

    def _run_async(self, coro):
        """Run an async coroutine synchronously using the dedicated loop."""
        assert self._nm_event_loop is not None, (
            "_run_async called but no N:M event loop initialised"
        )
        return self._nm_event_loop.run_until_complete(coro)

    def _do_propose_nm(
        self,
        active_req_ids: list[str],
        active_req_indices: list[int],
        seq_ids: torch.Tensor,
        k_accepted: torch.Tensor,
        bonus_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        hidden_states: torch.Tensor | None,
        extend_counts: torch.Tensor | None,
        extend_hidden_states: torch.Tensor | None,
        extend_token_ids: torch.Tensor | None,
        B_active: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """N:M speculation: group requests by server, send per-connector.

        Groups active requests by their assigned draft server index,
        sends one batched ``VerificationOutcome`` per server, then
        receives ``SpeculationResponse`` from each and reassembles
        the results in the original request order.

        Returns:
            ``(draft_tokens, draft_logits)`` tensors with shape
            ``[B_active, K]`` and ``[B_active, K, V]`` (or ``None``).
        """
        assert self.router is not None
        K = self.num_speculative_steps

        # Group active requests by their assigned server index.
        # Each request should already be assigned (via _prefill_new_requests).
        server_groups: dict[int, list[int]] = defaultdict(list)
        for j, rid in enumerate(active_req_ids):
            if rid in self.router.assignment:
                srv_idx = self.router.assignment[rid]
            else:
                # Request not yet assigned — assign now (shouldn't
                # normally happen since prefill assigns, but be safe).
                connector = self.router.assign(rid)
                srv_idx = self.router.assignment[rid]
            server_groups[srv_idx].append(j)

        # Allocate output tensors
        draft_toks_out = torch.zeros(
            B_active, K, dtype=torch.int64, device=self.device,
        )
        draft_logits_out: torch.Tensor | None = None

        # Send verification outcomes and receive speculation responses
        # per-connector.
        for srv_idx, local_indices in server_groups.items():
            connector = self.router.connectors[srv_idx]
            n = len(local_indices)
            idx_t = torch.tensor(
                local_indices, dtype=torch.int64, device=self.device,
            )

            # Slice tensors for this server's batch
            srv_seq_ids = seq_ids[idx_t]
            srv_k_accepted = k_accepted[idx_t]
            srv_bonus_tokens = bonus_tokens[idx_t]
            srv_temps = temperatures[idx_t]
            srv_hs = (
                hidden_states[idx_t] if hidden_states is not None else None
            )
            srv_ext_counts = (
                extend_counts[idx_t]
                if extend_counts is not None else None
            )
            srv_ext_hs = (
                extend_hidden_states[idx_t]
                if extend_hidden_states is not None else None
            )
            srv_ext_ids = (
                extend_token_ids[idx_t]
                if extend_token_ids is not None else None
            )

            try:
                # Send verification outcome and receive speculation in one call
                needs_logits = self.draft_logits is not None
                cache_hits, srv_draft_toks, srv_draft_logits = (
                    self._run_async(
                        connector.send_and_recv_speculation(
                            batch_size=n,
                            seq_ids=srv_seq_ids,
                            k_accepted=srv_k_accepted,
                            bonus_tokens=srv_bonus_tokens,
                            temperatures=srv_temps,
                            hidden_states=srv_hs,
                            aux_hidden_states=None,
                            extend_counts=srv_ext_counts,
                            extend_hidden_states=srv_ext_hs,
                            extend_token_ids=srv_ext_ids,
                            needs_logits=needs_logits,
                        )
                    )
                )

                # Map results back into the output tensors
                for local_j, global_j in enumerate(local_indices):
                    if local_j < srv_draft_toks.shape[0]:
                        draft_toks_out[global_j] = srv_draft_toks[local_j]

                if srv_draft_logits is not None:
                    if draft_logits_out is None:
                        draft_logits_out = torch.zeros(
                            B_active, K, self.vocab_size,
                            dtype=self.dtype, device=self.device,
                        )
                    for local_j, global_j in enumerate(local_indices):
                        if local_j < srv_draft_logits.shape[0]:
                            K_actual = min(
                                srv_draft_logits.shape[1],
                                draft_logits_out.shape[1],
                            )
                            draft_logits_out[global_j, :K_actual] = (
                                srv_draft_logits[local_j, :K_actual]
                            )

            except ConnectionError as e:
                logger.warning(
                    "N:M speculation from server %d failed with "
                    "ConnectionError: %s. Marking server unavailable "
                    "and reassigning requests.",
                    srv_idx, e,
                )
                self.router.handle_server_failure(srv_idx)
                # Requests assigned to this server get zeros (already
                # initialised to zero).

            except Exception as e:
                logger.warning(
                    "N:M speculation from server %d failed: %s. "
                    "Affected requests get zero draft tokens.",
                    srv_idx, e,
                )
                # Requests assigned to this server get zeros (already
                # initialised to zero).

        return draft_toks_out, draft_logits_out

    def _prefill_new_requests(
        self,
        input_batch,
        new_req_ids: list[str],
        last_hidden_states: torch.Tensor | None = None,
        aux_hidden_states: list[torch.Tensor] | None = None,
    ):
        """Send prefill data for new requests to the draft worker.

        Uses prompt tokens cached by cache_new_request_tokens() during
        add_requests(), avoiding UVA buffer reads.

        When ``needs_hidden_states`` is True and ``last_hidden_states``
        is provided, the target's last hidden state for each new
        request's final prompt token is extracted and sent alongside
        the prompt token IDs so the EAGLE head can begin speculation
        immediately after prefill.

        Supports both 1:1 (NCCL target interface) and N:M (DraftRouter)
        modes.
        """
        # Assign seq_ids to new requests (shared between 1:1 and N:M)
        for rid in new_req_ids:
            if rid not in self._disagg_req_to_seq_id:
                if self._disagg_free_seq_ids:
                    sid = self._disagg_free_seq_ids.pop()
                else:
                    sid = self._disagg_next_seq_id
                    self._disagg_next_seq_id += 1
                self._disagg_req_to_seq_id[rid] = sid

        if self.router is not None:
            self._prefill_new_requests_nm(
                input_batch, new_req_ids,
                last_hidden_states=last_hidden_states,
                aux_hidden_states=aux_hidden_states,
            )
        else:
            self._prefill_new_requests_1to1(
                input_batch, new_req_ids,
                last_hidden_states=last_hidden_states,
                aux_hidden_states=aux_hidden_states,
            )

    def _prefill_new_requests_nm(
        self,
        input_batch,
        new_req_ids: list[str],
        last_hidden_states: torch.Tensor | None = None,
        aux_hidden_states: list[torch.Tensor] | None = None,
    ):
        """N:M prefill: assign each new request to a draft server via
        the DraftRouter and send individual prefill requests per-connector.
        """
        assert self.router is not None

        for rid in new_req_ids:
            prompt_ids = self._pending_prompt_tokens.pop(rid, None)
            if prompt_ids is None or len(prompt_ids) == 0:
                logger.warning(
                    "Disagg N:M prefill req %s: no cached prompt tokens, "
                    "skipping", rid,
                )
                continue

            seq_id = self._disagg_req_to_seq_id[rid]

            # Assign request to a draft server
            try:
                connector = self.router.assign(rid)
            except RuntimeError:
                logger.error(
                    "No available draft servers for prefill of req %s", rid,
                )
                continue

            prompt_ids_t = torch.tensor(
                prompt_ids, dtype=torch.int64, device=self.device,
            )

            # Extract hidden states for this request's prompt tokens
            hs: torch.Tensor | None = None
            if (self.needs_hidden_states
                    and last_hidden_states is not None):
                try:
                    batch_idx = list(input_batch.req_ids).index(rid)
                except ValueError:
                    batch_idx = -1

                if batch_idx >= 0:
                    start = int(
                        input_batch.query_start_loc[batch_idx].item())
                    end = int(
                        input_batch.query_start_loc[batch_idx + 1].item())
                    if (aux_hidden_states is not None
                            and len(aux_hidden_states) > 0):
                        combined = torch.cat(
                            list(aux_hidden_states), dim=-1)
                        hs = combined[start:end]
                    else:
                        hs = last_hidden_states[start:end]

            try:
                self._run_async(
                    connector.send_prefill(
                        seq_id=seq_id,
                        prompt_token_ids=prompt_ids_t,
                        hidden_states=hs,
                    )
                )
            except Exception as e:
                logger.error(
                    "N:M prefill for req %s to server failed: %s",
                    rid, e,
                )
                continue

        self._disagg_prefilled_reqs.update(new_req_ids)

    def _prefill_new_requests_1to1(
        self,
        input_batch,
        new_req_ids: list[str],
        last_hidden_states: torch.Tensor | None = None,
        aux_hidden_states: list[torch.Tensor] | None = None,
    ):
        """Original 1:1 NCCL prefill path (unchanged logic)."""
        all_prompt_ids = []
        num_tokens_list = []
        new_seq_ids_list = []
        new_req_batch_indices: list[int] = []

        for rid in new_req_ids:
            prompt_ids = self._pending_prompt_tokens.pop(rid, None)
            if prompt_ids is None or len(prompt_ids) == 0:
                logger.warning(
                    "Disagg draft prefill req %s: no cached prompt tokens, skipping",
                    rid,
                )
                continue

            n_prompt = len(prompt_ids)

            # Always send ALL prompt token IDs (needed for draft prefill).
            # Hidden states may be partial (suffix only) if prefix caching
            # is active. The draft worker handles the mismatch.
            all_prompt_ids.extend(prompt_ids)
            num_tokens_list.append(n_prompt)
            new_seq_ids_list.append(self._disagg_req_to_seq_id[rid])

            try:
                batch_idx = list(input_batch.req_ids).index(rid)
                new_req_batch_indices.append(batch_idx)
            except ValueError:
                new_req_batch_indices.append(-1)

        if all_prompt_ids:
            input_ids_t = torch.tensor(
                all_prompt_ids, dtype=torch.int64,
                device=self.device,
            )
            num_tokens_t = torch.tensor(
                num_tokens_list, dtype=torch.int32,
                device=self.device,
            )
            new_seq_ids_t = torch.tensor(
                new_seq_ids_list, dtype=torch.int64,
                device=self.device,
            )

            # Extract hidden states for ALL prompt tokens of each new
            # request when the method requires them.  The standard EAGLE
            # speculator processes all prompt tokens through the EAGLE
            # head in a single "prefill" step, populating the EAGLE
            # head's KV cache for the full prompt context.  We replicate
            # this by sending all prompt hidden states to the draft
            # worker so it can run a full EAGLE prefill.
            hs = None
            if (self.needs_hidden_states
                    and last_hidden_states is not None
                    and new_req_batch_indices):
                hs_parts = []
                for idx, bi in enumerate(new_req_batch_indices):
                    if bi < 0:
                        continue
                    start = int(input_batch.query_start_loc[bi].item())
                    end = int(input_batch.query_start_loc[bi + 1].item())

                    if (aux_hidden_states is not None
                            and len(aux_hidden_states) > 0):
                        combined = torch.cat(
                            list(aux_hidden_states), dim=-1)
                        hs_parts.append(combined[start:end])
                    else:
                        hs_parts.append(last_hidden_states[start:end])
                if hs_parts:
                    hs = torch.cat(hs_parts, dim=0)

            try:
                self._target_interface.request_prefill(
                    input_ids_t, num_tokens_t,
                    seq_ids=new_seq_ids_t,
                    hidden_states=hs,
                )
            except Exception as e:
                logger.error(
                    "Disagg draft prefill send FAILED: %s", e,
                    exc_info=True,
                )
                # The PREFILL command was already sent. We must send
                # dummy prefill data so the draft worker doesn't hang.
                # Send a minimal 1-token prefill.
                try:
                    dummy_ids = torch.zeros(1, dtype=torch.int64,
                                            device=self.device)
                    dummy_ntok = torch.ones(1, dtype=torch.int32,
                                            device=self.device)
                    dummy_sids = new_seq_ids_t[:1]
                    self._target_interface.comm.send_prefill_data(
                        dummy_ids, dummy_ntok,
                        seq_ids=dummy_sids,
                    )
                except Exception:
                    pass
                return
            self._disagg_prefilled_reqs.update(new_req_ids)


class _DisaggDraftModelStub(nn.Module):
    """Minimal stub so model_runner doesn't crash on speculator.model."""

    def __init__(self):
        super().__init__()
