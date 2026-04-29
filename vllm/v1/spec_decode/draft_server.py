# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone Draft Server for disaggregated speculative decoding (N:M).

The ``DraftServer`` wraps the existing draft-side components
(``DraftModelRunner``, ``SpeculationCache``, ``OutcomePredictor``) in a
ZMQ ROUTER server that accepts connections from multiple Verify_Servers.

It reuses the core speculation logic from ``DisaggDraftWorker`` but
replaces the NCCL command loop with an async ZMQ server loop.  All
per-request state is namespaced by ``(verify_server_id, seq_id)`` to
prevent collisions when multiple Verify_Servers use overlapping
seq_id ranges.

This module does NOT modify the existing ``DisaggDraftWorker``.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from typing import TYPE_CHECKING, Any

import prometheus_client
import torch

from vllm.v1.spec_decode.draft_connector import (
    _dtype_to_str,
    _str_to_dtype,
    _tensor_to_bytes,
)
from vllm.v1.spec_decode.draft_data_models import (
    DraftCommand,
    FreeSeqRequest,
    PrefillRequest,
    SpeculationResponse,
    TensorRef,
    VerificationOutcome,
    decode,
    decode_command,
    encode,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Composite key type: (verify_server_id, seq_id)
RequestKey = tuple[str, int]


class DraftServerMetrics:
    """Prometheus metrics for the draft server side.

    Tracks batch size, generation latency, cache hit rate, evictions,
    connected verify server count, and active request count.
    """

    def __init__(self) -> None:
        self.draft_batch_size = prometheus_client.Gauge(
            name="vllm:draft_server_batch_size",
            documentation=(
                "Current batch size being processed by the draft server."
            ),
        )
        self.draft_generation_latency = prometheus_client.Histogram(
            name="vllm:draft_server_generation_latency_seconds",
            documentation=(
                "Latency (seconds) per speculation batch on the draft "
                "server."
            ),
            buckets=(
                0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0,
            ),
        )
        self.draft_cache_hit_rate = prometheus_client.Gauge(
            name="vllm:draft_server_cache_hit_rate",
            documentation=(
                "Rolling speculation cache hit rate on the draft server."
            ),
        )
        self.draft_eviction_count = prometheus_client.Counter(
            name="vllm:draft_server_eviction_total",
            documentation=(
                "Total number of request evictions due to verify server "
                "timeout."
            ),
        )
        self.draft_connected_verify_servers = prometheus_client.Gauge(
            name="vllm:draft_server_connected_verify_servers",
            documentation=(
                "Number of verify servers currently connected to the "
                "draft server."
            ),
        )
        self.draft_active_requests = prometheus_client.Gauge(
            name="vllm:draft_server_active_requests",
            documentation=(
                "Total active requests across all connected verify "
                "servers."
            ),
        )

        # Internal accumulators for rolling cache hit rate.
        self._total_lookups: int = 0
        self._total_hits: int = 0


class DraftServer:
    """Standalone draft server accepting requests from N verify servers.

    Reuses existing ``DraftModelRunner``, ``SpeculationCache``, and
    ``OutcomePredictor``.  Manages its own ZMQ ROUTER server loop.

    Args:
        vllm_config: Full vLLM configuration.
        bind_address: ZMQ address to bind the ROUTER socket
            (e.g. ``"tcp://*:50051"``).
    """

    def __init__(self, vllm_config: VllmConfig, bind_address: str) -> None:
        import zmq
        import zmq.asyncio

        from vllm.utils.network_utils import make_zmq_socket

        self.vllm_config = vllm_config
        self.bind_address = bind_address

        spec_config = vllm_config.speculative_config
        assert spec_config is not None

        self.K = spec_config.num_speculative_tokens
        self.vocab_size = spec_config.draft_model_config.get_vocab_size()
        self.target_vocab_size = vllm_config.model_config.get_vocab_size()
        self.dtype = vllm_config.model_config.dtype

        # Disagg-specific config
        self.fan_out = spec_config.disagg_fan_out
        self.saguaro_c = spec_config.disagg_saguaro_c
        self.jit_fallback = spec_config.disagg_jit_fallback
        self.needs_hidden_states = spec_config.disagg_needs_hidden_states

        # Determine device (draft server typically uses cuda:0)
        self.device = torch.device("cuda:0")

        max_batch_size = vllm_config.scheduler_config.max_num_seqs

        # ----- Existing draft-side components -----
        # Initialized by load_model() which must be called before serve().
        self.draft_model_runner: Any = None  # DraftModelRunner
        self.cache: Any = None  # SpeculationCache
        self.outcome_predictor: Any = None  # OutcomePredictor
        self.saguaro_sampler: Any = None  # SaguaroSampler

        # Pre-allocate hidden state buffer for EAGLE/EAGLE3/MTP methods
        self._target_hidden_states: torch.Tensor | None = None
        self._max_batch_size = max_batch_size

        # Per-round state tracking (mirrors DisaggDraftWorker)
        self._last_draft_tokens: torch.Tensor | None = None
        self._last_draft_logits: torch.Tensor | None = None
        self._last_bonus_tokens: torch.Tensor | None = None
        self._last_jit_prenorms: torch.Tensor | None = None
        self._glue_prenorm: torch.Tensor | None = None
        self._glue_logits: torch.Tensor | None = None
        self._round_base_lens: dict[int, int] = {}
        self._swap_states: dict[int, Any] = {}

        # EAGLE prefix cache (mirrors DisaggDraftWorker)
        # Key: hash of prompt token IDs tuple
        # Value: (block_ids list, seq_len after prefill)
        self._eagle_prefix_cache: dict[int, tuple[list[int], int]] = {}
        # Map seq_id → prompt token hash (for storing on completion)
        self._seq_prompt_hash: dict[int, int] = {}
        # Max entries to keep in the prefix cache
        self._eagle_prefix_cache_max = 64
        # Track prefill prenorm validity (mirrors DisaggDraftWorker)
        self._prefill_prenorm_valid = False

        # ----- NCCL process groups per verify server -----
        # Maps verify_server_id → (process_group, peer_rank)
        self._nccl_groups: dict[
            str, tuple[torch.distributed.ProcessGroup, int]
        ] = {}

        # Per-request tensor frames (ZMQ fallback, no NCCL).
        # Populated by the serve() loop and consumed by handlers.
        # Key: verify_server_id, Value: list of raw tensor frame bytes
        # from the current in-flight message.
        self._current_tensor_frames: list[bytes] = []
        self._current_tensor_idx: int = 0

        # Monotonic counter for unique buffer_id per tensor transfer
        self._buffer_counter = itertools.count()

        # ----- ZMQ ROUTER server -----
        self._ctx = zmq.asyncio.Context()
        self._socket = make_zmq_socket(
            self._ctx,
            self.bind_address,
            zmq.ROUTER,
            bind=True,
            router_handover=True,
            linger=1000,
        )

        # ----- Per-request state, keyed by (verify_server_id, seq_id) -----
        # Stores arbitrary per-request metadata needed by speculation logic.
        self._request_state: dict[RequestKey, dict[str, Any]] = {}

        # Track connected verify servers and their active request keys.
        self._verify_servers: dict[str, set[RequestKey]] = {}

        # ----- Timeout-based eviction -----
        # Monotonic timestamp of the last command received from each
        # verify server.  Used to detect disconnected servers.
        self._verify_server_last_seen: dict[str, float] = {}
        # Configurable timeout (seconds) before evicting a verify
        # server's requests when no commands have been received.
        self._eviction_timeout_s: float = 30.0

        # Server lifecycle flag
        self._running = False

        # ----- Metrics -----
        self.metrics = DraftServerMetrics()

        # ----- Multi-verify-server batching -----
        # Pending SPECULATE commands queued for batched processing.
        # Each entry is (verify_server_id, identity, VerificationOutcome).
        self._pending_speculations: list[
            tuple[str, bytes, VerificationOutcome]
        ] = []
        self._max_batch_size: int = max_batch_size
        # Timeout in seconds before flushing a partial batch.
        self._batch_timeout_s: float = 0.001  # 1ms default
        # Timestamp (monotonic) when the first item was added to the
        # current pending batch.  ``None`` when the queue is empty.
        self._batch_first_arrival: float | None = None

        logger.info(
            "DraftServer initialized, bind_address=%s, K=%d, "
            "fan_out=%d, needs_hidden_states=%s, max_batch_size=%d",
            self.bind_address,
            self.K,
            self.fan_out,
            self.needs_hidden_states,
            self._max_batch_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the draft model and initialize speculation components.

        Must be called before :meth:`serve`.  Mirrors the initialization
        done by ``DisaggDraftWorker.__init__`` + ``load_model()``.
        """
        from vllm.v1.worker.gpu.spec_decode.disagg_draft.draft_model_runner import (
            DraftModelRunner,
        )
        from vllm.v1.worker.gpu.spec_decode.disagg_draft.draft_worker import (
            OutcomePredictor,
            SaguaroSampler,
            SpeculationCache,
        )

        max_batch_size = self._max_batch_size

        # --- Initialize SpeculationCache ---
        self.cache = SpeculationCache(
            max_batch_size=max_batch_size,
            num_speculative_tokens=self.K,
            fan_out=self.fan_out,
            vocab_size=self.vocab_size,
            device=self.device,
            dtype=self.dtype,
        )

        # --- Initialize OutcomePredictor ---
        total_fan_out = self.fan_out * (self.K + 1)
        self.outcome_predictor = OutcomePredictor(
            num_speculative_tokens=self.K,
            total_fan_out=total_fan_out,
            acceptance_rate=0.65,
            power_law_exponent=1.5,
            device=self.device,
        )
        self.outcome_predictor.fan_out_list = [self.fan_out] * (self.K + 1)
        self.outcome_predictor.max_fan_out = self.fan_out

        # --- Initialize SaguaroSampler ---
        self.saguaro_sampler = SaguaroSampler(
            saguaro_c=self.saguaro_c,
            fan_out=self.fan_out,
            device=self.device,
        )

        # --- Initialize DraftModelRunner (loads model + allocates KV cache) ---
        logger.info("DraftServer loading draft model...")
        self.draft_model_runner = DraftModelRunner(
            vllm_config=self.vllm_config,
            device=self.device,
        )
        self.draft_model_runner.load_model()
        logger.info(
            "DraftServer model loaded: K=%d, vocab=%d, device=%s",
            self.K,
            self.vocab_size,
            self.device,
        )

    async def serve(self) -> None:
        """Main loop: accept commands from multiple verify servers.

        Runs until an EXIT command is received or :meth:`shutdown` is
        called.  Each incoming ZMQ message is a multipart frame:
        ``[identity, b"", payload]`` where *identity* is the
        verify_server_id set by the DEALER socket on the connector side.

        Uses a polling loop with a short timeout so that partial
        speculation batches are flushed even when no new messages
        arrive within ``_batch_timeout_s``.
        """
        import zmq

        self._running = True
        logger.info("DraftServer serving on %s", self.bind_address)

        poller = zmq.asyncio.Poller()
        poller.register(self._socket, zmq.POLLIN)

        while self._running:
            # Compute poll timeout: if we have pending speculations,
            # wait at most until the batch timeout expires.
            if self._pending_speculations and self._batch_first_arrival is not None:
                elapsed = time.monotonic() - self._batch_first_arrival
                remaining_ms = max(
                    0, (self._batch_timeout_s - elapsed) * 1000
                )
                poll_timeout_ms = int(remaining_ms)
            else:
                # No pending batch — block until a message arrives
                # (use a long timeout to avoid busy-waiting).
                poll_timeout_ms = 1000

            try:
                events = dict(await poller.poll(timeout=poll_timeout_ms))
            except Exception:
                if not self._running:
                    break
                logger.exception("DraftServer poll error")
                continue

            if self._socket in events:
                try:
                    frames = await self._socket.recv_multipart(
                        flags=zmq.NOBLOCK
                    )
                except Exception:
                    if not self._running:
                        break
                    logger.exception("DraftServer recv error")
                    continue

                # ZMQ ROUTER frames: [identity, metadata, tensor0, ...]
                # DEALER sends [metadata, t0, ...] → ROUTER prepends identity.
                # There is NO empty delimiter when using send_multipart.
                if len(frames) < 2:
                    logger.warning(
                        "DraftServer received malformed message "
                        "with %d frames (need ≥2), skipping",
                        len(frames),
                    )
                else:
                    identity = frames[0]
                    metadata_frame = frames[1]
                    tensor_frames = frames[2:]  # may be empty (NCCL path)
                    verify_server_id = identity.decode(
                        "utf-8", errors="replace"
                    )

                    try:
                        command = decode_command(metadata_frame)
                    except Exception:
                        logger.exception(
                            "DraftServer failed to decode command from %s",
                            verify_server_id,
                        )
                        continue

                    # Store tensor frames for this message so handlers
                    # can consume them via _zmq_nccl_recv().
                    self._current_tensor_frames = list(tensor_frames)
                    self._current_tensor_idx = 0

                    await self._dispatch(
                        verify_server_id, identity, command
                    )

            # After processing any incoming message (or on poll
            # timeout), check whether the pending batch should be
            # flushed due to timeout.
            await self._maybe_flush_batch()

            # Check for verify servers that have timed out and evict
            # their requests.
            self._check_evictions()

    async def shutdown(self) -> None:
        """Gracefully stop the server loop and release resources."""
        self._running = False
        self._cleanup()
        logger.info("DraftServer shut down.")

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    async def _dispatch(
        self,
        verify_server_id: str,
        identity: bytes,
        command: DraftCommand,
    ) -> None:
        """Route a decoded command to the appropriate handler.

        SPECULATE commands are queued for batched processing.  All
        other commands flush any pending speculations first, then
        execute immediately.

        Updates the last-seen timestamp for the verify server on
        every command to support timeout-based eviction.
        """
        # Track activity for timeout-based eviction.
        self._verify_server_last_seen[verify_server_id] = time.monotonic()

        # Update connected server count metric.
        self.metrics.draft_connected_verify_servers.set(
            len(self._verify_servers)
        )

        cmd = command.command.upper()

        if cmd == "SPECULATE":
            outcome = decode(command.payload, VerificationOutcome)
            # When using ZMQ tensor transport (no NCCL), process immediately
            # to avoid the tensor frames being consumed by the serve() loop.
            if verify_server_id not in self._nccl_groups:
                await self._handle_speculation(
                    verify_server_id, identity, outcome
                )
                return
            self._enqueue_speculation(
                verify_server_id, identity, outcome
            )
            # Trigger batch processing if the queue is full.
            if len(self._pending_speculations) >= self._max_batch_size:
                await self._process_batched_speculation()
            return

        # For non-SPECULATE commands, flush any pending speculations
        # first so that ordering is preserved.
        if self._pending_speculations:
            await self._process_batched_speculation()

        if cmd == "PREFILL":
            prefill = decode(command.payload, PrefillRequest)
            await self._handle_prefill(
                verify_server_id, identity, prefill
            )

        elif cmd == "FREE_SEQ":
            free_req = decode(command.payload, FreeSeqRequest)
            await self._handle_free_seq(
                verify_server_id, identity, free_req
            )

        elif cmd == "EXIT":
            logger.info(
                "DraftServer received EXIT from %s", verify_server_id
            )
            await self._handle_exit(verify_server_id, identity)

        elif cmd == "HEALTHCHECK":
            await self._handle_healthcheck(verify_server_id, identity)

        else:
            logger.warning(
                "DraftServer received unknown command '%s' from %s",
                cmd,
                verify_server_id,
            )

    # ------------------------------------------------------------------
    # Multi-verify-server batching
    # ------------------------------------------------------------------

    def _enqueue_speculation(
        self,
        verify_server_id: str,
        identity: bytes,
        outcome: VerificationOutcome,
    ) -> None:
        """Add a SPECULATE command to the pending batch queue."""
        if not self._pending_speculations:
            self._batch_first_arrival = time.monotonic()
        self._pending_speculations.append(
            (verify_server_id, identity, outcome)
        )
        logger.debug(
            "DraftServer queued SPECULATE from %s, "
            "pending=%d/%d",
            verify_server_id,
            len(self._pending_speculations),
            self._max_batch_size,
        )

    async def _maybe_flush_batch(self) -> None:
        """Flush the pending batch if the timeout has expired."""
        if not self._pending_speculations:
            return
        if self._batch_first_arrival is None:
            return
        elapsed = time.monotonic() - self._batch_first_arrival
        if elapsed >= self._batch_timeout_s:
            await self._process_batched_speculation()

    async def _process_batched_speculation(self) -> None:
        """Process all pending SPECULATE commands as a single batch.

        For each pending entry, delegates to the existing
        ``_handle_speculation`` which handles NCCL tensor receives,
        speculation logic, and response sending.  After processing,
        the pending queue is cleared.

        The batch is capped at ``_max_batch_size``.  If the queue
        exceeds that limit (shouldn't happen normally since we flush
        on reaching the limit), only the first ``_max_batch_size``
        entries are processed and the rest remain queued.
        """
        if not self._pending_speculations:
            return

        # Take up to max_batch_size entries from the queue.
        batch_size = min(
            len(self._pending_speculations), self._max_batch_size
        )
        batch = self._pending_speculations[:batch_size]
        self._pending_speculations = self._pending_speculations[batch_size:]

        # Reset the batch timer.  If there are still pending items,
        # start a new timer for the remaining entries.
        if self._pending_speculations:
            self._batch_first_arrival = time.monotonic()
        else:
            self._batch_first_arrival = None

        logger.debug(
            "DraftServer processing batched speculation, "
            "batch_size=%d, remaining=%d",
            len(batch),
            len(self._pending_speculations),
        )

        # Process each entry in the batch.  Each call to
        # _handle_speculation handles its own NCCL receives and
        # ZMQ response sends, so the verify servers are served
        # in order within the batch.
        for verify_server_id, identity, outcome in batch:
            await self._handle_speculation(
                verify_server_id, identity, outcome
            )

    # ------------------------------------------------------------------
    # Timeout-based eviction
    # ------------------------------------------------------------------

    def _check_evictions(self) -> None:
        """Evict requests for verify servers that have timed out.

        For each verify server tracked in ``_verify_servers``, checks
        whether the time since the last received command exceeds
        ``_eviction_timeout_s``.  If so, frees all KV cache blocks and
        speculation cache entries for that server's requests, then
        removes the server from tracking.

        Called periodically from the ``serve()`` loop on each poll
        cycle.
        """
        if not self._verify_servers:
            return

        now = time.monotonic()
        timed_out_servers: list[str] = []

        for vs_id in list(self._verify_servers):
            last_seen = self._verify_server_last_seen.get(vs_id)
            if last_seen is None:
                # Never received a command — skip (shouldn't happen
                # in practice since _dispatch sets last_seen).
                continue
            if now - last_seen > self._eviction_timeout_s:
                timed_out_servers.append(vs_id)

        for vs_id in timed_out_servers:
            keys = list(self._verify_servers.get(vs_id, set()))
            logger.warning(
                "DraftServer evicting %d requests for timed-out "
                "verify server %s (last seen %.1fs ago)",
                len(keys),
                vs_id,
                now - self._verify_server_last_seen.get(vs_id, now),
            )

            # Free KV cache and per-request state for each request.
            runner = self.draft_model_runner
            for _vs_id, seq_id in keys:
                # Clear per-round state
                self._round_base_lens.pop(seq_id, None)
                self._swap_states.pop(seq_id, None)
                self._seq_prompt_hash.pop(seq_id, None)

                if runner is not None:
                    runner.free_blocks(seq_id)

                self._request_state.pop((_vs_id, seq_id), None)

            # Increment eviction counter.
            self.metrics.draft_eviction_count.inc(len(keys))

            # Remove the verify server from tracking.
            self._verify_servers.pop(vs_id, None)
            self._verify_server_last_seen.pop(vs_id, None)

            # Also remove any pending speculations from this server.
            self._pending_speculations = [
                (sid, ident, outcome)
                for sid, ident, outcome in self._pending_speculations
                if sid != vs_id
            ]

    # ------------------------------------------------------------------
    # Request namespacing helpers
    # ------------------------------------------------------------------

    def _make_key(self, verify_server_id: str, seq_id: int) -> RequestKey:
        """Create a composite key for per-request state."""
        return (verify_server_id, seq_id)

    def _register_request(
        self, verify_server_id: str, seq_id: int
    ) -> RequestKey:
        """Register a new request and return its composite key."""
        key = self._make_key(verify_server_id, seq_id)
        if key not in self._request_state:
            self._request_state[key] = {}
        # Track under the verify server
        if verify_server_id not in self._verify_servers:
            self._verify_servers[verify_server_id] = set()
        self._verify_servers[verify_server_id].add(key)
        # Update active request count metric.
        self.metrics.draft_active_requests.set(len(self._request_state))
        return key

    def _unregister_request(
        self, verify_server_id: str, seq_id: int
    ) -> None:
        """Remove a request's state and tracking."""
        key = self._make_key(verify_server_id, seq_id)
        self._request_state.pop(key, None)
        server_keys = self._verify_servers.get(verify_server_id)
        if server_keys is not None:
            server_keys.discard(key)
            if not server_keys:
                del self._verify_servers[verify_server_id]
        # Update active request count metric.
        self.metrics.draft_active_requests.set(len(self._request_state))

    def _get_request_state(self, key: RequestKey) -> dict[str, Any]:
        """Get per-request state, creating if absent."""
        if key not in self._request_state:
            self._request_state[key] = {}
        return self._request_state[key]

    # ------------------------------------------------------------------
    # Tensor transport helpers
    # ------------------------------------------------------------------

    def _zmq_nccl_recv(
        self,
        verify_server_id: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Receive a tensor — from NCCL PG or from current message frames."""
        if verify_server_id not in self._nccl_groups:
            # ZMQ path: consume next frame from the current message.
            idx = self._current_tensor_idx
            self._current_tensor_idx += 1
            if idx < len(self._current_tensor_frames):
                buf = self._current_tensor_frames[idx]
                recv_dtype = torch.float32 if dtype == torch.bfloat16 else dtype
                tensor = torch.frombuffer(
                    bytearray(buf), dtype=recv_dtype
                ).reshape(shape).to(dtype=dtype, device=self.device)
                return tensor
            logger.warning(
                "DraftServer: tensor frame %d missing (have %d frames), "
                "returning zeros for shape=%s dtype=%s",
                idx, len(self._current_tensor_frames), shape, dtype,
            )
            return torch.zeros(shape, dtype=dtype, device=self.device)

        pg, peer_rank = self._nccl_groups[verify_server_id]
        tensor = torch.empty(shape, dtype=dtype, device=self.device)
        pg.recv([tensor], peer_rank, 0).wait()
        return tensor

    def _zmq_nccl_send(
        self,
        verify_server_id: str,
        tensor: torch.Tensor,
        pending_frames: list[bytes],
    ) -> None:
        """Send a tensor — via NCCL PG or append to pending_frames list."""
        if verify_server_id not in self._nccl_groups:
            pending_frames.append(_tensor_to_bytes(tensor))
            return
        pg, peer_rank = self._nccl_groups[verify_server_id]
        pg.send([tensor.contiguous().to(self.device)], peer_rank, 0).wait()

    def _make_tensor_ref(self, tensor: torch.Tensor) -> TensorRef:
        """Build a TensorRef for an outgoing tensor."""
        return TensorRef(
            shape=tuple(tensor.shape),
            dtype=_dtype_to_str(tensor.dtype),
            buffer_id=str(next(self._buffer_counter)),
            nbytes=tensor.nelement() * tensor.element_size(),
        )

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

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
           deterministic send order in ``ZmqNcclDraftConnector``).
        2. Reset ``_seq_lens`` and run glue decode for EAGLE methods.
        3. Cache lookup via ``SpeculationCache.lookup``.
        4. Hybrid swap+JIT: cache hits use cached tokens, misses run
           ``_eagle_jit_speculate`` or ``_jit_speculate``.
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

        try:
            await self._handle_speculation_inner(
                verify_server_id, identity, outcome
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

    async def _handle_speculation_inner(
        self,
        verify_server_id: str,
        identity: bytes,
        outcome: VerificationOutcome,
    ) -> None:
        """Core speculation logic, separated for error handling."""
        B = outcome.batch_size
        _spec_start = time.monotonic()

        # Record current batch size.
        self.metrics.draft_batch_size.set(B)

        # ---- Step 1: Receive tensor payloads ----
        # Tensor frames are already in self._current_tensor_frames,
        # populated by the serve() loop before calling _dispatch.
        # Order must match ZmqNcclDraftConnector.send_verification_outcome:
        # seq_ids, k_accepted, bonus_tokens, [temperatures],
        # [hidden_states], [aux_hidden_states], [extend_counts],
        # [extend_hidden_states], [extend_token_ids]
        seq_ids = self._zmq_nccl_recv(
            verify_server_id,
            outcome.seq_ids_ref.shape,
            _str_to_dtype(outcome.seq_ids_ref.dtype),
        )
        k_accepted = self._zmq_nccl_recv(
            verify_server_id,
            outcome.k_accepted_ref.shape,
            _str_to_dtype(outcome.k_accepted_ref.dtype),
        )
        bonus_tokens = self._zmq_nccl_recv(
            verify_server_id,
            outcome.bonus_tokens_ref.shape,
            _str_to_dtype(outcome.bonus_tokens_ref.dtype),
        )

        temperatures: torch.Tensor | None = None
        if outcome.temperatures_ref is not None:
            temperatures = self._zmq_nccl_recv(
                verify_server_id,
                outcome.temperatures_ref.shape,
                _str_to_dtype(outcome.temperatures_ref.dtype),
            )

        hidden_states: torch.Tensor | None = None
        if outcome.hidden_states_ref is not None:
            hidden_states = self._zmq_nccl_recv(
                verify_server_id,
                outcome.hidden_states_ref.shape,
                _str_to_dtype(outcome.hidden_states_ref.dtype),
            )

        aux_hidden_states: torch.Tensor | None = None
        if outcome.aux_hidden_states_ref is not None:
            aux_hidden_states = self._zmq_nccl_recv(
                verify_server_id,
                outcome.aux_hidden_states_ref.shape,
                _str_to_dtype(outcome.aux_hidden_states_ref.dtype),
            )

        extend_counts: torch.Tensor | None = None
        if outcome.extend_counts_ref is not None:
            extend_counts = self._zmq_nccl_recv(
                verify_server_id,
                outcome.extend_counts_ref.shape,
                _str_to_dtype(outcome.extend_counts_ref.dtype),
            )

        extend_hidden_states: torch.Tensor | None = None
        if outcome.extend_hidden_states_ref is not None:
            extend_hidden_states = self._zmq_nccl_recv(
                verify_server_id,
                outcome.extend_hidden_states_ref.shape,
                _str_to_dtype(outcome.extend_hidden_states_ref.dtype),
            )

        extend_token_ids: torch.Tensor | None = None
        if outcome.extend_token_ids_ref is not None:
            extend_token_ids = self._zmq_nccl_recv(
                verify_server_id,
                outcome.extend_token_ids_ref.shape,
                _str_to_dtype(outcome.extend_token_ids_ref.dtype),
            )

        # Store hidden states for EAGLE/EAGLE3/MTP methods
        if self.needs_hidden_states and hidden_states is not None:
            if self._target_hidden_states is None:
                self._target_hidden_states = torch.zeros(
                    self._max_batch_size,
                    hidden_states.shape[-1],
                    dtype=hidden_states.dtype,
                    device=self.device,
                )
            self._target_hidden_states[:B] = hidden_states

        # ---- Step 1b: Reset _seq_lens and run glue decode ----
        runner = self.draft_model_runner
        if runner is not None:
            seq_ids_list = seq_ids.tolist()
            k_accepted_list = k_accepted.tolist()

            for i, sid in enumerate(seq_ids_list):
                swap_rec = self._swap_states.get(sid)
                if swap_rec is not None and getattr(
                    swap_rec, "last_round_was_swap", False
                ):
                    correct_len = (
                        getattr(swap_rec, "swap_prefix_len", 0)
                        + 1
                        + int(k_accepted_list[i])
                    )
                    runner._seq_lens[sid] = correct_len
                elif sid in self._round_base_lens:
                    k_acc = int(k_accepted_list[i])
                    if self.needs_hidden_states:
                        runner._seq_lens[sid] = self._round_base_lens[sid]
                    else:
                        runner._seq_lens[sid] = (
                            self._round_base_lens[sid] + 1 + k_acc
                        )

            # Pre-allocate blocks for the full speculation round
            for sid in seq_ids_list:
                cur_len = runner._seq_lens.get(sid, 0)
                needed = cur_len + self.K + self.K + 2
                runner.ensure_blocks(sid, needed)

            # Run glue decode for EAGLE methods
            if self.needs_hidden_states and hidden_states is not None:
                if extend_counts is None:
                    extend_counts = torch.zeros(
                        B, dtype=torch.int64, device=self.device
                    )
                self._run_glue_decode(
                    B=B,
                    seq_ids=seq_ids,
                    k_accepted=k_accepted,
                    bonus_tokens=bonus_tokens,
                    hidden_states=hidden_states,
                    extend_counts=extend_counts,
                    extend_hidden_states=extend_hidden_states,
                    extend_token_ids=extend_token_ids,
                )
            else:
                self._glue_prenorm = None
                self._glue_logits = None

            # Save base lens BEFORE any JIT or swap modifies _seq_lens
            for sid in seq_ids_list:
                self._round_base_lens[sid] = runner._seq_lens.get(sid, 0)
        else:
            seq_ids_list = seq_ids.tolist()

        # ---- Step 2: Cache lookup ----
        cached_tokens, cached_logits, cache_hits, _cached_hs = (
            self.cache.lookup(
                seq_ids=seq_ids,
                k_accepted=k_accepted,
                bonus_tokens=bonus_tokens,
            )
        )

        # ---- Step 3: Hybrid swap+JIT ----
        num_hits = int(cache_hits.sum().item())
        hit_mask = cache_hits.bool()
        miss_mask = ~hit_mask

        # Update rolling cache hit rate metric.
        self.metrics._total_lookups += B
        self.metrics._total_hits += num_hits
        if self.metrics._total_lookups > 0:
            self.metrics.draft_cache_hit_rate.set(
                self.metrics._total_hits / self.metrics._total_lookups
            )

        # Pre-allocate output tensors
        draft_tokens = torch.zeros(
            B, self.K, dtype=torch.int64, device=self.device
        )
        draft_logits = torch.zeros(
            B, self.K, self.vocab_size, dtype=self.dtype, device=self.device
        )

        # --- Handle cache hits ---
        used_swap_for_hits = False
        if num_hits > 0 and cached_logits is not None:
            if self.needs_hidden_states:
                # EAGLE: no block swapping needed. Return cached tokens.
                draft_tokens[hit_mask] = cached_tokens[hit_mask]
                draft_logits[hit_mask] = cached_logits[hit_mask]
                used_swap_for_hits = True
            else:
                # Standalone: swap dedicated block tables into main
                hit_tables, hit_prefix_lens = (
                    self.cache.get_hit_block_tables(cache_hits)
                )
                if (
                    hit_tables is not None
                    and hit_prefix_lens is not None
                    and runner is not None
                ):
                    hit_seq_ids = seq_ids[hit_mask]
                    owned, displaced = runner.swap_block_tables(
                        seq_ids=hit_seq_ids,
                        branch_block_tables=hit_tables,
                        prefix_lens=hit_prefix_lens,
                        K=self.K,
                    )
                    for blocks in owned.values():
                        runner.exclude_from_dedicated(blocks)
                    if displaced:
                        runner._free_list.extend(displaced)

                    hit_indices = hit_mask.nonzero(as_tuple=True)[0]
                    for compact_i, idx in enumerate(hit_indices):
                        i = int(idx.item())
                        sid = seq_ids_list[i]
                        prefix_len = int(
                            hit_prefix_lens[compact_i].item()
                        )
                        runner._seq_lens[sid] = prefix_len + self.K

                    draft_tokens[hit_mask] = cached_tokens[hit_mask]
                    draft_logits[hit_mask] = cached_logits[hit_mask]
                    used_swap_for_hits = True

        # --- Handle cache misses: JIT only on misses ---
        B_miss = int(miss_mask.sum().item())
        if B_miss > 0:
            miss_seq_ids = seq_ids[miss_mask]
            miss_bonus = bonus_tokens[miss_mask]
            miss_temps = (
                temperatures[miss_mask]
                if temperatures is not None
                else None
            )

            if self.needs_hidden_states and runner is not None:
                miss_hidden = self._target_hidden_states[
                    miss_mask.nonzero(as_tuple=True)[0]
                ]
                glue_prenorm = self._glue_prenorm
                miss_glue_prenorm = None
                if glue_prenorm is not None:
                    miss_glue_prenorm = glue_prenorm[
                        miss_mask.nonzero(as_tuple=True)[0]
                    ]
                glue_logits = self._glue_logits
                miss_glue_logits = None
                if glue_logits is not None:
                    miss_glue_logits = glue_logits[
                        miss_mask.nonzero(as_tuple=True)[0]
                    ]
                jit_tokens, jit_logits = self._eagle_jit_speculate(
                    miss_seq_ids,
                    miss_bonus,
                    B_miss=B_miss,
                    temperatures=miss_temps,
                    hidden_states=miss_hidden,
                    glue_prenorm=miss_glue_prenorm,
                    glue_logits=miss_glue_logits,
                )
            else:
                jit_tokens, jit_logits = self._jit_speculate(
                    miss_seq_ids,
                    miss_bonus,
                    B_miss=B_miss,
                    temperatures=miss_temps,
                )

            draft_tokens[miss_mask] = jit_tokens
            if jit_logits is not None:
                draft_logits[miss_mask] = jit_logits

            # Expand JIT prenorms to full batch size for cache building
            if (
                self.needs_hidden_states
                and self._last_jit_prenorms is not None
            ):
                full_prenorms = torch.zeros(
                    B,
                    self.K,
                    self._last_jit_prenorms.shape[-1],
                    dtype=self._last_jit_prenorms.dtype,
                    device=self.device,
                )
                full_prenorms[miss_mask] = self._last_jit_prenorms
                self._last_jit_prenorms = full_prenorms
        else:
            if self.needs_hidden_states:
                self._last_jit_prenorms = None

        if not used_swap_for_hits:
            for sid in seq_ids_list:
                self._swap_states[sid] = {}

        # Store for _build_next_cache
        self._last_draft_tokens = draft_tokens.clone()
        self._last_draft_logits = draft_logits.clone()
        self._last_bonus_tokens = bonus_tokens.clone()

        # ---- Step 4: Send SpeculationResponse ----
        # Only send draft_logits if the verify server requested them
        send_logits = outcome.needs_logits
        await self._send_speculation_response(
            verify_server_id, identity, cache_hits, draft_tokens,
            draft_logits if send_logits else None,
        )

        # Record generation latency for this speculation batch.
        self.metrics.draft_generation_latency.observe(
            time.monotonic() - _spec_start
        )

        # ---- Step 5: Build speculation cache for NEXT round ----
        if runner is not None and not self.needs_hidden_states:
            saved_seq_lens = dict(runner._seq_lens)
        self._build_next_cache(B, seq_ids)
        if runner is not None and not self.needs_hidden_states:
            runner._seq_lens = saved_seq_lens

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    async def _send_speculation_response(
        self,
        verify_server_id: str,
        identity: bytes,
        cache_hits: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor | None,
    ) -> None:
        """Send a SpeculationResponse back to the verify server."""
        cache_hits_ref = self._make_tensor_ref(cache_hits)
        draft_tokens_ref = self._make_tensor_ref(draft_tokens)
        draft_logits_ref = (
            self._make_tensor_ref(draft_logits)
            if draft_logits is not None else None
        )

        resp = SpeculationResponse(
            cache_hits_ref=cache_hits_ref,
            draft_tokens_ref=draft_tokens_ref,
            draft_logits_ref=draft_logits_ref,
        )
        resp_bytes = encode(resp)

        if verify_server_id in self._nccl_groups:
            await self._socket.send_multipart([identity, resp_bytes])
            pending: list[bytes] = []
            self._zmq_nccl_send(verify_server_id, cache_hits, pending)
            self._zmq_nccl_send(verify_server_id, draft_tokens, pending)
            if draft_logits is not None:
                self._zmq_nccl_send(verify_server_id, draft_logits, pending)
        else:
            pending = []
            self._zmq_nccl_send(verify_server_id, cache_hits, pending)
            self._zmq_nccl_send(verify_server_id, draft_tokens, pending)
            if draft_logits is not None:
                self._zmq_nccl_send(verify_server_id, draft_logits, pending)
            await self._socket.send_multipart(
                [identity, resp_bytes] + pending
            )

    async def _send_fallback_speculation(
        self,
        verify_server_id: str,
        identity: bytes,
        batch_size: int,
    ) -> None:
        """Send a fallback (all-zeros) SpeculationResponse on error.

        Ensures the verify server does not hang waiting for a response.
        """
        B = max(batch_size, 1)
        cache_hits = torch.zeros(B, dtype=torch.int64, device=self.device)
        draft_tokens = torch.zeros(
            B, self.K, dtype=torch.int64, device=self.device
        )
        draft_logits = torch.zeros(
            B, self.K, self.vocab_size, dtype=self.dtype, device=self.device
        )

        try:
            await self._send_speculation_response(
                verify_server_id,
                identity,
                cache_hits,
                draft_tokens,
                None,  # no logits in fallback
            )
        except Exception:
            logger.exception(
                "DraftServer failed to send fallback response to %s",
                verify_server_id,
            )

    # ------------------------------------------------------------------
    # Speculation logic (delegates to existing components)
    # ------------------------------------------------------------------

    def _eagle_jit_speculate(
        self,
        seq_ids: torch.Tensor,
        bonus_tokens: torch.Tensor,
        B_miss: int,
        temperatures: torch.Tensor | None,
        hidden_states: torch.Tensor,
        glue_prenorm: torch.Tensor | None = None,
        glue_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """JIT speculation for EAGLE/EAGLE3/MTP methods (cache misses).

        Delegates to ``DraftModelRunner.eagle_sequential_speculate``.
        Falls back to random tokens when the draft model is not loaded.
        """
        runner = self.draft_model_runner
        if runner is not None and runner._model_loaded:
            positions = torch.tensor(
                [
                    runner._seq_lens.get(int(sid), 0)
                    for sid in seq_ids.tolist()
                ],
                dtype=torch.long,
                device=self.device,
            )

            processed_hs = hidden_states
            if hasattr(runner.model, "combine_hidden_states"):
                processed_hs = runner.model.combine_hidden_states(
                    hidden_states
                )

            tokens, logits_out, jit_prenorms = (
                runner.eagle_sequential_speculate(
                    recovery_tokens=bonus_tokens,
                    positions=positions,
                    seq_ids=seq_ids,
                    num_steps=self.K,
                    hidden_states=processed_hs,
                    temperatures=temperatures,
                    glue_prenorm=glue_prenorm,
                    glue_logits=glue_logits,
                )
            )

            self._last_jit_prenorms = jit_prenorms

            if self.target_vocab_size < self.vocab_size:
                tokens = tokens.clamp(max=self.target_vocab_size - 1)

            return tokens, logits_out

        # Random token fallback
        tokens = torch.randint(
            0,
            self.vocab_size,
            (B_miss, self.K),
            device=self.device,
            dtype=torch.int64,
        )
        tokens[:, 0] = bonus_tokens
        logits = torch.zeros(
            B_miss,
            self.K,
            self.vocab_size,
            dtype=self.dtype,
            device=self.device,
        ).uniform_()
        return tokens, logits

    def _jit_speculate(
        self,
        seq_ids: torch.Tensor,
        recovery_tokens: torch.Tensor,
        B_miss: int,
        temperatures: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """JIT speculation for standalone draft models (cache misses).

        Delegates to ``DraftModelRunner.sequential_speculate``.
        Falls back to random tokens when the draft model is not loaded.
        """
        runner = self.draft_model_runner
        if runner is not None and runner._model_loaded:
            positions = torch.tensor(
                [
                    runner._seq_lens.get(int(sid), 0)
                    for sid in seq_ids.tolist()
                ],
                dtype=torch.long,
                device=self.device,
            )
            tokens, logits = runner.sequential_speculate(
                recovery_tokens=recovery_tokens,
                positions=positions,
                seq_ids=seq_ids,
                num_steps=self.K,
                temperature=temperatures,
                saguaro_sampler=(
                    self.saguaro_sampler
                    if self.saguaro_c is not None
                    else None
                ),
            )
            if self.target_vocab_size < self.vocab_size:
                tokens = tokens.clamp(max=self.target_vocab_size - 1)
            return tokens, logits

        # Random token fallback
        tokens = torch.randint(
            0,
            self.vocab_size,
            (B_miss, self.K),
            device=self.device,
            dtype=torch.int64,
        )
        tokens[:, 0] = recovery_tokens
        logits = torch.zeros(
            B_miss,
            self.K,
            self.vocab_size,
            dtype=self.dtype,
            device=self.device,
        ).uniform_()
        return tokens, logits

    def _run_glue_decode(
        self,
        B: int,
        seq_ids: torch.Tensor,
        k_accepted: torch.Tensor,
        bonus_tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        extend_counts: torch.Tensor,
        extend_hidden_states: torch.Tensor | None,
        extend_token_ids: torch.Tensor | None,
    ) -> None:
        """Run glue decode to fill EAGLE KV cache gaps.

        Delegates to the same logic as
        ``DisaggDraftWorker._run_glue_decode``.  When the target accepts
        draft tokens (k_accepted > 0), the EAGLE head's KV cache has
        gaps at positions that were never written.  This method fills
        those gaps by running a single batched forward pass with
        extend + recovery tokens.

        Stores the EAGLE model's prenorm output at the recovery token
        position in ``self._glue_prenorm`` for use by JIT step 0.
        """
        runner = self.draft_model_runner
        if runner is None or not runner._model_loaded:
            return

        seq_ids_list = seq_ids.tolist()
        ext_counts = (
            extend_counts.tolist()
            if extend_counts is not None
            else [0] * B
        )

        # Compute per-sequence token counts: n_ext + 1 (recovery)
        seqlens_q: list[int] = []
        for i in range(B):
            n_ext = int(ext_counts[i])
            seqlens_q.append(n_ext + 1)

        total_tokens = sum(seqlens_q)
        if total_tokens == 0:
            return

        # Build packed input_ids and hidden_states
        fused_ids = torch.zeros(
            total_tokens, dtype=torch.int64, device=self.device
        )
        fused_hs = torch.zeros(
            total_tokens,
            hidden_states.shape[-1],
            dtype=hidden_states.dtype,
            device=self.device,
        )

        offset = 0
        for i in range(B):
            n_ext = int(ext_counts[i])
            if (
                n_ext > 0
                and extend_token_ids is not None
                and extend_hidden_states is not None
            ):
                fused_ids[offset : offset + n_ext] = extend_token_ids[
                    i, :n_ext
                ]
                fused_hs[offset : offset + n_ext] = extend_hidden_states[
                    i, :n_ext
                ]
            fused_ids[offset + n_ext] = bonus_tokens[i]
            fused_hs[offset + n_ext] = hidden_states[i]
            offset += n_ext + 1

        # Project hidden states through combine_hidden_states (fc)
        if hasattr(runner.model, "combine_hidden_states"):
            processed_hs = runner.model.combine_hidden_states(fused_hs)
        else:
            processed_hs = fused_hs

        # Build positions
        positions = torch.zeros(
            total_tokens, dtype=torch.long, device=self.device
        )
        expanded_seq_ids: list[int] = []
        offset = 0
        for i in range(B):
            sid = seq_ids_list[i]
            n_ext = int(ext_counts[i])
            n_q = n_ext + 1
            start_pos = runner._seq_lens.get(sid, 0)
            positions[offset : offset + n_q] = (
                torch.arange(n_q, device=self.device) + start_pos
            )
            expanded_seq_ids.extend([sid] * n_q)
            offset += n_q

        # Compute slot mapping and block tables
        slot_mapping = runner._compute_slot_mapping(
            positions, expanded_seq_ids
        )
        block_tables = runner._get_block_table_tensor(seq_ids)

        # Build FlashAttention metadata
        seq_lens_list = []
        for i in range(B):
            sid = seq_ids_list[i]
            n_ext = int(ext_counts[i])
            n_q = n_ext + 1
            start_pos = runner._seq_lens.get(sid, 0)
            seq_lens_list.append(start_pos + n_q)
        seq_lens_t = torch.tensor(
            seq_lens_list, dtype=torch.int32, device=self.device
        )
        max_seq_len = int(seq_lens_t.max().item())

        seqlens_q_t = torch.tensor(
            seqlens_q, dtype=torch.int32, device=self.device
        )
        max_query_len = max(seqlens_q)
        query_start_loc = torch.zeros(
            B + 1, dtype=torch.int32, device=self.device
        )
        torch.cumsum(seqlens_q_t, dim=0, out=query_start_loc[1:])

        from vllm.forward_context import BatchDescriptor, set_forward_context

        attn_metadata = runner._build_flash_attn_metadata(
            num_tokens=total_tokens,
            seq_lens_tensor=seq_lens_t,
            max_seq_len=max_seq_len,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            block_table=block_tables,
            slot_mapping=slot_mapping,
        )
        slot_mapping_dict = runner._build_slot_mapping_dict(slot_mapping)

        batch_descriptor = BatchDescriptor(num_tokens=total_tokens)
        with set_forward_context(
            attn_metadata=attn_metadata,
            vllm_config=runner._draft_vllm_config,
            num_tokens=total_tokens,
            slot_mapping=slot_mapping_dict,
            batch_descriptor=batch_descriptor,
        ):
            output = runner.model(
                input_ids=fused_ids,
                positions=positions,
                hidden_states=processed_hs,
            )

        # Extract prenorm and logits at recovery token position
        seqlens_q_long = torch.tensor(
            seqlens_q, dtype=torch.long, device=self.device
        )
        last_indices = torch.cumsum(seqlens_q_long, dim=0) - 1

        if runner.method != "mtp":
            last_hs, out_hs = output
            self._glue_prenorm = out_hs[last_indices]
            sample_hs = last_hs[last_indices]
            if hasattr(runner.model, "compute_logits"):
                self._glue_logits = runner.model.compute_logits(sample_hs)
            elif hasattr(runner.model, "lm_head"):
                self._glue_logits = runner.model.lm_head(sample_hs)
            else:
                self._glue_logits = None
            if self._glue_logits is not None:
                self._glue_logits = self._glue_logits[
                    :, : runner.vocab_size
                ]
        else:
            self._glue_prenorm = output[last_indices]
            self._glue_logits = None

        # Update _seq_lens to reflect the glue decode
        for i in range(B):
            sid = seq_ids_list[i]
            n_ext = int(ext_counts[i])
            n_q = n_ext + 1
            start_pos = runner._seq_lens.get(sid, 0)
            runner._seq_lens[sid] = start_pos + n_q

    def _build_next_cache(
        self,
        batch_size: int,
        seq_ids: torch.Tensor,
    ) -> None:
        """Pre-compute speculation cache for the NEXT round.

        Delegates to the existing ``SpeculationCache`` and
        ``DraftModelRunner`` tree decode logic, matching the
        ``DisaggDraftWorker._build_next_cache`` flow.
        """
        if self.cache is not None:
            self.cache.reset()

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
        F = self.fan_out

        max_branches = 504
        if B * (K + 1) * F > max_branches:
            F = max(1, max_branches // (B * (K + 1)))
        N = B * (K + 1) * F
        if N > max_branches:
            return

        draft_tokens = self._last_draft_tokens
        draft_logits = self._last_draft_logits
        rec_tokens = self._last_bonus_tokens

        seq_ids_list = seq_ids.tolist()

        if self.needs_hidden_states:
            self._build_eagle_cache(
                B, K, F, N, seq_ids, seq_ids_list, runner,
                draft_tokens, draft_logits, rec_tokens,
            )
        else:
            self._build_standalone_cache(
                B, K, F, N, seq_ids, seq_ids_list, runner,
                draft_tokens, draft_logits, rec_tokens,
            )

    def _build_eagle_cache(
        self,
        B: int, K: int, F: int, N: int,
        seq_ids: torch.Tensor,
        seq_ids_list: list[int],
        runner: Any,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor,
        rec_tokens: torch.Tensor,
    ) -> None:
        """Build speculation cache for EAGLE methods.

        Follows the same fused glue decode → fork → tree decode flow
        as ``DisaggDraftWorker._build_eagle_cache``.
        """
        saved_seq_lens = dict(runner._seq_lens)

        Kp1 = K + 1
        jit_prenorms = self._last_jit_prenorms
        hidden_states = self._target_hidden_states

        # Step 1: Fused glue decode (recovery + spec tokens)
        prenorm_flat: torch.Tensor | None = None
        if hidden_states is None:
            glue_logits_kp1 = draft_logits[:, -1]
            outcome_logits = torch.cat(
                [draft_logits, glue_logits_kp1.unsqueeze(1)], dim=1
            )
        else:
            total_tokens = B * Kp1
            fused_ids = torch.zeros(
                total_tokens, dtype=torch.int64, device=self.device
            )
            fused_ids_2d = fused_ids.view(B, Kp1)
            fused_ids_2d[:, 0] = rec_tokens
            fused_ids_2d[:, 1:] = draft_tokens

            if hasattr(runner.model, "combine_hidden_states"):
                rec_hs = runner.model.combine_hidden_states(
                    hidden_states[:B]
                )
                eagle_hs_dim = rec_hs.shape[-1]
            else:
                rec_hs = hidden_states[:B]
                eagle_hs_dim = hidden_states.shape[-1]

            fused_hs = torch.zeros(
                total_tokens, eagle_hs_dim,
                dtype=rec_hs.dtype, device=self.device,
            )
            fused_hs_2d = fused_hs.view(B, Kp1, eagle_hs_dim)
            fused_hs_2d[:, 0] = rec_hs
            if jit_prenorms is not None and jit_prenorms.shape[0] >= B:
                fused_hs_2d[:, 1:] = jit_prenorms[:B, :K]
            else:
                fused_hs_2d[:, 1:] = rec_hs.unsqueeze(1).expand(
                    B, K, eagle_hs_dim
                )

            positions = torch.zeros(
                total_tokens, dtype=torch.long, device=self.device
            )
            positions_2d = positions.view(B, Kp1)
            for i in range(B):
                sid = seq_ids_list[i]
                base = self._round_base_lens.get(sid, 0)
                positions_2d[i] = (
                    torch.arange(Kp1, device=self.device) + base
                )

            expanded_seq_ids_flat: list[int] = []
            for i in range(B):
                expanded_seq_ids_flat.extend([seq_ids_list[i]] * Kp1)
            slot_mapping = runner._compute_slot_mapping(
                positions, expanded_seq_ids_flat
            )
            block_tables = runner._get_block_table_tensor(seq_ids)

            seqlens_q = torch.full(
                (B,), Kp1, dtype=torch.int32, device=self.device
            )
            query_start_loc = torch.zeros(
                B + 1, dtype=torch.int32, device=self.device
            )
            torch.cumsum(seqlens_q, dim=0, out=query_start_loc[1:])

            seq_lens_list = []
            for i in range(B):
                sid = seq_ids_list[i]
                base = self._round_base_lens.get(sid, 0)
                seq_lens_list.append(base + Kp1)
            seq_lens_t = torch.tensor(
                seq_lens_list, dtype=torch.int32, device=self.device
            )
            max_seq_len = int(seq_lens_t.max().item())

            from vllm.forward_context import (
                BatchDescriptor,
                set_forward_context,
            )

            attn_metadata = runner._build_flash_attn_metadata(
                num_tokens=total_tokens,
                seq_lens_tensor=seq_lens_t,
                max_seq_len=max_seq_len,
                max_query_len=Kp1,
                query_start_loc=query_start_loc,
                block_table=block_tables,
                slot_mapping=slot_mapping,
            )
            slot_mapping_dict = runner._build_slot_mapping_dict(
                slot_mapping
            )

            batch_descriptor = BatchDescriptor(num_tokens=total_tokens)
            with set_forward_context(
                attn_metadata=attn_metadata,
                vllm_config=runner._draft_vllm_config,
                num_tokens=total_tokens,
                slot_mapping=slot_mapping_dict,
                batch_descriptor=batch_descriptor,
            ):
                output = runner.model(
                    input_ids=fused_ids,
                    positions=positions,
                    hidden_states=fused_hs,
                )

            if runner.method != "mtp":
                last_hs_flat, prenorm_flat = output
            else:
                last_hs_flat = output
                prenorm_flat = output

            if hasattr(runner.model, "compute_logits"):
                glue_logits_flat = runner.model.compute_logits(
                    last_hs_flat
                )
            elif hasattr(runner.model, "lm_head"):
                glue_logits_flat = runner.model.lm_head(last_hs_flat)
            else:
                glue_logits_flat = torch.matmul(
                    last_hs_flat,
                    runner.model.get_input_embeddings().weight.T,
                )
            glue_logits_flat = glue_logits_flat[:, : self.vocab_size]
            outcome_logits = glue_logits_flat.view(B, Kp1, -1)

        # Step 2: Fork bonus candidates
        outcome_tokens = torch.cat(
            [rec_tokens.unsqueeze(1), draft_tokens], dim=1
        )
        masked_logits = outcome_logits.clone()
        masked_logits[:, :-1, :] = masked_logits[:, :-1, :].scatter(
            dim=2,
            index=outcome_tokens[:, 1:].unsqueeze(2),
            value=float("-inf"),
        )
        _, topk_indices = torch.topk(masked_logits, F, dim=-1)

        batch_ids_grid = (
            torch.arange(B, device=self.device)
            .view(B, 1, 1)
            .expand(B, Kp1, F)
        )
        k_pos_grid = (
            torch.arange(Kp1, device=self.device, dtype=torch.int64)
            .view(1, Kp1, 1)
            .expand(B, Kp1, F)
        )

        k_positions = k_pos_grid.reshape(-1)
        bonus_candidates = topk_indices.reshape(-1)
        entry_batch_ids = batch_ids_grid.reshape(-1)

        # Step 3: Ensure blocks for tree decode
        branches_per_seq = Kp1 * F
        max_tree_pos = Kp1 + branches_per_seq * K + K
        for sid in seq_ids_list:
            base = self._round_base_lens.get(sid, 0)
            runner.ensure_blocks(sid, base + max_tree_pos + 1)

        base_lens_t = torch.tensor(
            [
                self._round_base_lens.get(int(seq_ids[b].item()), 0)
                for b in range(B)
            ],
            dtype=torch.int64,
            device=self.device,
        )

        branch_within_seq = (
            torch.arange(
                branches_per_seq, device=self.device, dtype=torch.int64
            )
            .unsqueeze(0)
            .expand(B, -1)
            .reshape(-1)
        )

        tree_start = (
            base_lens_t[entry_batch_ids] + Kp1 + branch_within_seq * K
        )

        seq_ids_expanded = seq_ids[entry_batch_ids]
        tree_block_tables = runner._block_table_gpu[
            seq_ids_expanded.to(torch.int64)
        ].contiguous()

        # Step 4: Tree decode (K steps)
        all_tokens = torch.zeros(
            N, K, dtype=torch.int64, device=self.device
        )
        all_logits = torch.zeros(
            N, K, self.vocab_size, dtype=self.dtype, device=self.device
        )
        current_ids = bonus_candidates.clone()
        max_context_hint = int(tree_start.max().item()) + K + 1

        # Initialize per-branch hidden states from glue prenorms
        if hidden_states is not None and prenorm_flat is not None:
            glue_prenorm_kp1 = prenorm_flat.view(B, Kp1, -1)
            branch_hidden_states = glue_prenorm_kp1[
                entry_batch_ids, k_positions
            ]
        else:
            branch_hidden_states = None

        for depth in range(K):
            tree_positions = tree_start + depth
            context_lens = tree_positions + 1

            if branch_hidden_states is not None:
                logits, branch_hidden_states = (
                    runner.eagle_tree_decode_step(
                        input_ids=current_ids,
                        positions=tree_positions,
                        seq_lens=context_lens,
                        seq_ids_expanded=seq_ids_expanded,
                        block_tables=tree_block_tables,
                        hidden_states=branch_hidden_states,
                        max_seq_len_hint=max_context_hint,
                    )
                )
            else:
                logits = runner.tree_decode_step(
                    input_ids=current_ids,
                    positions=tree_positions,
                    seq_lens=context_lens,
                    seq_ids_expanded=seq_ids_expanded,
                    block_tables=tree_block_tables,
                    max_seq_len_hint=max_context_hint,
                )

            all_logits[:, depth] = logits
            next_tokens = logits.argmax(dim=-1)
            all_tokens[:, depth] = next_tokens
            current_ids = next_tokens

        # Step 5: Populate cache
        self.cache.populate(
            seq_ids=seq_ids[entry_batch_ids],
            k_positions=k_positions,
            bonus_tokens=bonus_candidates,
            draft_tokens=all_tokens,
            draft_logits=all_logits,
            branch_block_tables=tree_block_tables,
            prefix_lens=tree_start,
        )

        # Restore _seq_lens
        runner._seq_lens = saved_seq_lens

    def _build_standalone_cache(
        self,
        B: int, K: int, F: int, N: int,
        seq_ids: torch.Tensor,
        seq_ids_list: list[int],
        runner: Any,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor,
        rec_tokens: torch.Tensor,
    ) -> None:
        """Build speculation cache for standalone draft models.

        Uses dedicated blocks with KV copy, matching
        ``DisaggDraftWorker._build_standalone_cache``.
        """
        runner.recycle_dedicated_blocks()

        # Glue decode for K+1th logits
        glue_logits = runner.glue_decode(
            tokens=draft_tokens[:, -1], seq_ids=seq_ids
        )

        post_glue_lens = {
            sid: runner._seq_lens.get(sid, 0) for sid in seq_ids_list
        }

        Kp1 = K + 1
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
        _, topk_indices = torch.topk(masked_logits, F, dim=-1)

        batch_ids_grid = (
            torch.arange(B, device=self.device)
            .view(B, 1, 1)
            .expand(B, Kp1, F)
        )
        k_pos_grid = (
            torch.arange(Kp1, device=self.device, dtype=torch.int64)
            .view(1, Kp1, 1)
            .expand(B, Kp1, F)
        )

        k_positions = k_pos_grid.reshape(-1)
        bonus_candidates = topk_indices.reshape(-1)
        entry_batch_ids = batch_ids_grid.reshape(-1)

        # Bounds check for dedicated blocks
        bs = runner.block_size
        M = runner.max_num_blocks
        blocks_per_branch = (K + bs) // bs + 1
        total_needed = N * blocks_per_branch
        available = (
            (runner.num_kv_blocks - runner._next_free_block)
            + len(runner._free_list)
        )
        if available < total_needed:
            for sid in seq_ids_list:
                if sid in post_glue_lens:
                    runner._seq_lens[sid] = post_glue_lens[sid] - 1
            return

        dedicated_blocks = [
            runner._alloc_one_block() for _ in range(total_needed)
        ]
        runner.reserve_dedicated_blocks(dedicated_blocks)

        # Build per-branch block tables
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

        # Copy KV from parent to dedicated blocks
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
                layer_kv[:, dst_flat] = layer_kv[:, src_flat]

        # Tree decode (K steps)
        seq_ids_expanded = seq_ids[entry_batch_ids]
        all_tokens = torch.zeros(
            N, K, dtype=torch.int64, device=self.device
        )
        all_logits = torch.zeros(
            N, K, self.vocab_size, dtype=self.dtype, device=self.device
        )
        current_ids = bonus_candidates.clone()

        max_prefix = int(prefix_lens.max().item())
        max_context_hint = max_prefix + K + 1

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

        self.cache.populate(
            seq_ids=seq_ids[entry_batch_ids],
            k_positions=k_positions,
            bonus_tokens=bonus_candidates,
            draft_tokens=all_tokens,
            draft_logits=all_logits,
            branch_block_tables=branch_block_tables,
            prefix_lens=prefix_lens,
        )

        # Restore _seq_lens (undo glue's +1)
        for sid in seq_ids_list:
            if sid in post_glue_lens:
                runner._seq_lens[sid] = post_glue_lens[sid] - 1

    # ------------------------------------------------------------------
    # Prefill and free_seq command handlers
    # ------------------------------------------------------------------

    async def _handle_prefill(
        self,
        verify_server_id: str,
        identity: bytes,
        prefill: PrefillRequest,
    ) -> None:
        """Handle PREFILL command for a new sequence.

        Receives tensor payloads via NCCL (matching the send order in
        ``ZmqNcclDraftConnector.send_prefill``), registers the request
        under the composite key, and delegates to the existing prefill
        logic from ``DisaggDraftWorker._handle_prefill``.

        For standalone draft models, runs a standard prefill to populate
        the draft model's KV cache with prompt tokens.

        For EAGLE/EAGLE3/MTP methods (``needs_hidden_states=True``),
        also processes hidden states through the EAGLE head so the
        EAGLE KV cache is properly initialised.
        """
        key = self._register_request(verify_server_id, prefill.seq_id)
        logger.debug(
            "DraftServer PREFILL from %s, seq_id=%d, key=%s",
            verify_server_id,
            prefill.seq_id,
            key,
        )

        # ---- Step 1: Receive tensor payloads ----
        # Tensor frames are already in self._current_tensor_frames.
        # Order matches ZmqNcclDraftConnector.send_prefill:
        # prompt_token_ids, then optionally hidden_states
        prompt_token_ids = self._zmq_nccl_recv(
            verify_server_id,
            prefill.prompt_token_ids_ref.shape,
            _str_to_dtype(prefill.prompt_token_ids_ref.dtype),
        )

        hidden_states: torch.Tensor | None = None
        if prefill.hidden_states_ref is not None:
            hidden_states = self._zmq_nccl_recv(
                verify_server_id,
                prefill.hidden_states_ref.shape,
                _str_to_dtype(prefill.hidden_states_ref.dtype),
            )

        # ---- Step 2: Delegate to existing prefill logic ----
        seq_id = prefill.seq_id
        B = 1  # Prefill is per-sequence
        num_tokens = torch.tensor(
            [prompt_token_ids.shape[0]],
            dtype=torch.int64,
            device=self.device,
        )
        seq_ids = torch.tensor(
            [seq_id], dtype=torch.int64, device=self.device
        )

        logger.info(
            "DraftServer prefill: seq_id=%d, num_tokens=%d, "
            "needs_hidden_states=%s, has_hidden_states=%s",
            seq_id,
            int(num_tokens[0].item()),
            self.needs_hidden_states,
            hidden_states is not None,
        )

        runner = self.draft_model_runner
        if runner is None or not runner._model_loaded:
            return

        if self.needs_hidden_states:
            # EAGLE/EAGLE3/MTP: run EAGLE prefill with available
            # hidden states.
            if hidden_states is not None:
                processed_hs = hidden_states
                if hasattr(runner.model, "combine_hidden_states"):
                    processed_hs = runner.model.combine_hidden_states(
                        hidden_states
                    )

                total_tokens = int(prompt_token_ids.shape[0])
                hs_tokens = int(processed_hs.shape[0])

                if hs_tokens >= total_tokens:
                    # Full prefill — all hidden states available
                    prefill_ids = prompt_token_ids
                    prefill_hs = processed_hs
                    prefill_ntok = num_tokens
                    pos_offsets = None
                else:
                    # Prefix cache hit — only suffix hidden states.
                    prompt_hash = hash(tuple(prompt_token_ids.tolist()))
                    cached = self._eagle_prefix_cache.get(prompt_hash)
                    if cached is not None:
                        cached_blocks, cached_seq_len = cached
                        sid = int(seq_ids[0].item())
                        n = int(num_tokens[0].item())
                        runner.allocate_blocks(sid, n + 256)
                        # Copy KV data from cached blocks
                        new_blocks = runner._block_tables.get(sid, [])
                        n_copy = min(len(cached_blocks), len(new_blocks))
                        if n_copy > 0 and runner.kv_caches is not None:
                            src = torch.tensor(
                                cached_blocks[:n_copy],
                                dtype=torch.int64,
                                device=self.device,
                            )
                            dst = torch.tensor(
                                new_blocks[:n_copy],
                                dtype=torch.int64,
                                device=self.device,
                            )
                            for layer_kv in runner.kv_caches:
                                layer_kv[:, dst] = layer_kv[:, src]
                        runner._seq_lens[sid] = n - 1
                        self._seq_prompt_hash[sid] = prompt_hash
                        logger.info(
                            "DraftServer EAGLE prefix cache HIT: "
                            "hash=%d, copied %d blocks, seq_len=%d",
                            prompt_hash,
                            n_copy,
                            cached_seq_len,
                        )
                    else:
                        # No cached EAGLE KV — allocate and set _seq_lens
                        for i in range(B):
                            sid = int(seq_ids[i].item())
                            n = int(num_tokens[i].item())
                            runner.allocate_blocks(sid, n + 256)
                            runner._seq_lens[sid] = n - 1
                        for i in range(B):
                            sid = int(seq_ids[i].item())
                            self._seq_prompt_hash[sid] = prompt_hash
                        logger.info(
                            "DraftServer EAGLE prefix cache MISS: "
                            "hash=%d, skipping EAGLE prefill.",
                            prompt_hash,
                        )
                    prefill_ids = None  # signal to skip

                if prefill_ids is not None:
                    try:
                        runner.eagle_prefill(
                            input_ids=prefill_ids,
                            num_tokens_per_seq=prefill_ntok,
                            seq_ids=seq_ids,
                            hidden_states=prefill_hs,
                            position_offsets=pos_offsets,
                        )
                        torch.cuda.synchronize(self.device)

                        self._glue_prenorm = None
                        self._glue_logits = None
                        self._prefill_prenorm_valid = False

                        logger.info(
                            "DraftServer EAGLE prefill OK: "
                            "seq_lens=%s, hs shape=%s, "
                            "prefix_cached=%s",
                            {
                                int(s): runner._seq_lens.get(int(s), -1)
                                for s in seq_ids.tolist()
                            },
                            processed_hs.shape,
                            hs_tokens < total_tokens,
                        )
                        prompt_hash = hash(
                            tuple(prompt_token_ids.tolist())
                        )
                        for i in range(B):
                            sid = int(seq_ids[i].item())
                            self._seq_prompt_hash[sid] = prompt_hash
                    except Exception as e:
                        logger.warning(
                            "DraftServer EAGLE prefill failed: %s", e
                        )
                        self._glue_prenorm = None
                        self._glue_logits = None
                        for i, sid in enumerate(seq_ids.tolist()):
                            n = int(num_tokens[i].item())
                            try:
                                runner.allocate_blocks(int(sid), n)
                            except Exception:
                                pass
                            runner._seq_lens[int(sid)] = 0
            else:
                # No hidden states — just allocate blocks
                self._glue_prenorm = None
                self._glue_logits = None
                for i, sid in enumerate(seq_ids.tolist()):
                    n = int(num_tokens[i].item())
                    try:
                        runner.allocate_blocks(int(sid), n)
                    except (RuntimeError, ValueError) as e:
                        logger.warning(
                            "DraftServer EAGLE prefill block alloc "
                            "failed for seq %d: %s",
                            sid,
                            e,
                        )
                    runner._seq_lens[int(sid)] = 0
        else:
            # Standalone draft model: run standard prefill.
            self._glue_prenorm = None
            self._glue_logits = None
            try:
                runner.prefill(
                    input_ids=prompt_token_ids,
                    num_tokens_per_seq=num_tokens,
                    seq_ids=seq_ids,
                )
            except (RuntimeError, ValueError) as e:
                logger.warning("DraftServer prefill failed: %s", e)
                return

        # Clear stale round base lengths for freshly prefilled sequences.
        for sid in seq_ids.tolist():
            self._round_base_lens.pop(int(sid), None)
            self._swap_states.pop(int(sid), None)

    async def _handle_free_seq(
        self,
        verify_server_id: str,
        identity: bytes,
        free_req: FreeSeqRequest,
    ) -> None:
        """Handle FREE_SEQ command to release resources.

        Receives the seq_ids tensor via NCCL (matching the send order
        in ``ZmqNcclDraftConnector.send_free_seq``), then for each
        seq_id: frees KV cache blocks, clears speculation cache entries,
        and unregisters the request.

        Before freeing, caches the EAGLE KV blocks for prefix reuse
        (mirrors ``DisaggDraftWorker._handle_free_seq``).
        """
        logger.debug(
            "DraftServer FREE_SEQ from %s", verify_server_id
        )

        # ---- Step 1: Receive seq_ids tensor ----
        # Tensor frames are already in self._current_tensor_frames.
        seq_ids = self._zmq_nccl_recv(
            verify_server_id,
            free_req.seq_ids_ref.shape,
            _str_to_dtype(free_req.seq_ids_ref.dtype),
        )

        # ---- Step 2: Free resources for each sequence ----
        runner = self.draft_model_runner
        freed = 0
        for sid in seq_ids.tolist():
            sid = int(sid)

            # Clear per-round state
            self._round_base_lens.pop(sid, None)
            self._swap_states.pop(sid, None)

            if runner is not None:
                # Cache EAGLE KV blocks before freeing
                prompt_hash = self._seq_prompt_hash.pop(sid, None)
                if (
                    prompt_hash is not None
                    and prompt_hash not in self._eagle_prefix_cache
                ):
                    blocks = runner._block_tables.get(sid)
                    seq_len = runner._seq_lens.get(sid)
                    if blocks and seq_len:
                        self._eagle_prefix_cache[prompt_hash] = (
                            list(blocks),
                            seq_len,
                        )
                        # Remove from _block_tables so free_blocks
                        # doesn't recycle them — they're now owned
                        # by the prefix cache.
                        runner._block_tables.pop(sid, None)
                        runner._seq_lens.pop(sid, None)
                        if sid < runner._block_table_gpu.shape[0]:
                            runner._block_table_gpu[sid].zero_()
                        # Evict oldest if cache is full
                        if (
                            len(self._eagle_prefix_cache)
                            > self._eagle_prefix_cache_max
                        ):
                            oldest_key = next(
                                iter(self._eagle_prefix_cache)
                            )
                            evicted_blocks, _ = (
                                self._eagle_prefix_cache.pop(oldest_key)
                            )
                            runner._free_list.extend(evicted_blocks)
                        freed += 1
                        # Unregister the request
                        self._unregister_request(verify_server_id, sid)
                        continue
                runner.free_blocks(sid)
                freed += 1

            # Unregister the request
            self._unregister_request(verify_server_id, sid)

        if freed:
            logger.debug(
                "DraftServer freed %d sequences for %s.",
                freed,
                verify_server_id,
            )

    async def _handle_exit(
        self, verify_server_id: str, identity: bytes
    ) -> None:
        """Handle EXIT command from a verify server.

        Cleans up all state for the disconnecting verify server.
        """
        keys = list(self._verify_servers.get(verify_server_id, set()))
        for key in keys:
            self._request_state.pop(key, None)
        self._verify_servers.pop(verify_server_id, None)
        logger.info(
            "DraftServer cleaned up %d requests for exiting "
            "verify server %s",
            len(keys),
            verify_server_id,
        )

    async def _handle_healthcheck(
        self, verify_server_id: str, identity: bytes
    ) -> None:
        """Handle HEALTHCHECK command — respond with an ack.

        Sends a simple acknowledgement back to the verify server so it
        can confirm the draft server is alive.
        """
        try:
            ack = encode(DraftCommand(command="HEALTHCHECK_ACK", payload=b""))
            await self._socket.send_multipart([identity, ack])
        except Exception:
            logger.exception(
                "DraftServer failed to send HEALTHCHECK_ACK to %s",
                verify_server_id,
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        """Release ZMQ resources."""
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
        if self._ctx is not None:
            try:
                self._ctx.term()
            except Exception:
                pass
            self._ctx = None
        self._request_state.clear()
        self._verify_servers.clear()
        self._verify_server_last_seen.clear()
        self._pending_speculations.clear()
        self._batch_first_arrival = None
