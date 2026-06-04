# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone draft server for disaggregated speculative decoding (N:M).

``DraftServer`` wraps a ``DraftModelRunner`` and ``SpeculationCache`` in a
ZMQ ROUTER that accepts SPECULATE / PREFILL / FREE_SEQ commands from
multiple verify servers. It is launched as its own process via
``vllm serve --draft-server ...``; verify servers connect to it over the
network via ``ZmqDraftConnector``.

Each verify server's external seq_id space is remapped to an internal
seq_id range that is globally unique across all connected verify servers,
so KV cache and speculation-cache state stays isolated.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
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

        # Cross-VS SPECULATE merge counters (Option A).
        self.draft_speculate_total = prometheus_client.Counter(
            name="vllm:draft_server_speculate_total",
            documentation=(
                "Total SPECULATEs processed (a merged round counts as 2)."
            ),
        )
        self.draft_speculate_merged = prometheus_client.Counter(
            name="vllm:draft_server_speculate_merged_total",
            documentation=(
                "SPECULATEs that participated in a cross-VS merged "
                "batch (counts each VS individually, so a successful "
                "2-VS merge increments by 2)."
            ),
        )


class DraftServer:
    """Standalone draft server accepting requests from N verify servers.

    Reuses existing ``DraftModelRunner``, ``SpeculationCache``, and
    ``OutcomePredictor``.  Manages its own ZMQ ROUTER server loop.

    Args:
        vllm_config: Full vLLM configuration.
        bind_address: ZMQ address to bind the ROUTER socket
            (e.g. ``"tcp://*:50051"``).
    """

    # Cap on the total number of cache-build branches per round.
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

        # Parallel fanout: use single-pass MTP-style drafting when the
        # model supports it (configured via speculative_config).
        self._use_parallel_fanout = spec_config.disagg_parallel_fanout
        self._mtp_token_id: int = spec_config.disagg_mtp_token_id

        # SSD §4.3 fast-backup: on cache miss, return zero drafts to
        # the verifier (saves the full ~17 ms K-step JIT) and let
        # cache_build seed real cache entries by running ONE
        # glue_decode per miss row using the bonus token. Per-round
        # miss_mask is set by the speculate path and consumed by
        # cache_build's glue_decode to decide which input token
        # (bonus vs draft_tokens[:, -1]) to feed each row.
        self._last_miss_mask: torch.Tensor | None = None

        # Determine device (use current CUDA device, set by entrypoint)
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")

        max_batch_size = vllm_config.scheduler_config.max_num_seqs

        # ----- Existing draft-side components -----
        # Initialized by load_model() which must be called before serve().
        self.draft_model_runner: Any = None  # DraftModelRunner
        self.cache: Any = None  # SpeculationCache
        self.outcome_predictor: Any = None  # OutcomePredictor
        self.saguaro_sampler: Any = None  # SaguaroSampler

        self._max_batch_size = max_batch_size

        # Per-round state tracking.
        self._last_draft_tokens: torch.Tensor | None = None
        self._last_draft_logits: torch.Tensor | None = None
        self._last_bonus_tokens: torch.Tensor | None = None
        self._round_base_lens: dict[int, int] = {}
        self._swap_states: dict[int, Any] = {}

        # Last speculation seq_ids — stored by _handle_speculation_inner,
        # consumed by _handle_speculation for post-response cache building.
        self._last_spec_seq_ids: torch.Tensor | None = None

        # In-flight background cache build. Awaited before the next
        # SPECULATE begins work on the runner (runner state and the
        # SpeculationCache are mutated by _build_next_cache, so new
        # handlers must wait). Returns the serve loop to recv_multipart
        # immediately, overlapping ZMQ recv/decode for the next message
        # with the GPU-bound cache build on the default stream.
        self._inflight_cache_build: asyncio.Task | None = None

        # Per-request tensor frames.
        # Populated by the serve() loop and consumed by handlers.
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

        # ----- Cross-VS SPECULATE merging (Option A) -----
        # The serve loop opportunistically peeks for additional pending
        # SPECULATEs from other VSes after receiving the first, and runs
        # them all as a single merged batch. This collapses the N× draft
        # serialization that would otherwise happen under multi-VS load.
        # Always-on; no-op when only one VS is connected (the peek is
        # guarded by ``len(self._verify_servers) >= 2`` in the serve loop).
        # DISAGG_MERGE_PEEK_MS controls how long to wait (in ms) for a
        # second pending SPECULATE from a different VS before giving up
        # and processing the first message alone.
        try:
            self._merge_peek_timeout_ms: int = int(
                os.environ.get("DISAGG_MERGE_PEEK_MS", "1")
            )
        except ValueError:
            self._merge_peek_timeout_ms = 1

        # ----- torch.profiler hooks (toggled via start/stop_profile) -----
        # Active when VLLM_DRAFT_TORCH_PROFILER_DIR is set in the env and
        # start_profile() has been called. Traces are written on stop.
        self._profiler: Any = None
        self._profiler_dir: str | None = None

        # ----- Metrics -----
        self.metrics = DraftServerMetrics()

        # ----- Seq ID remapping for multi-verify-server isolation -----
        # Each verify server assigns its own seq_ids starting from 0.
        # To avoid collisions in the DraftModelRunner's KV cache and
        # block tables, we remap (verify_server_id, external_seq_id)
        # to a unique internal_seq_id.
        self._ext_to_int_seq: dict[tuple[str, int], int] = {}
        self._int_to_ext_seq: dict[int, tuple[str, int]] = {}
        self._next_internal_seq_id: int = 0
        self._free_internal_seq_ids: list[int] = []

        logger.info(
            "DraftServer initialized, bind_address=%s, K=%d, "
            "fan_out=%d, max_batch_size=%d",
            self.bind_address,
            self.K,
            self.fan_out,
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
        from vllm.v1.worker.gpu.spec_decode.disagg_draft.outcome_predictor import (
            OutcomePredictor,
        )
        from vllm.v1.worker.gpu.spec_decode.disagg_draft.saguaro_sampling import (
            SaguaroSampler,
        )
        from vllm.v1.worker.gpu.spec_decode.disagg_draft.speculation_cache import (
            SpeculationCache,
        )

        max_batch_size = self._max_batch_size

        # --- Initialize SpeculationCache ---
        # max_verify_servers sizes the per-VS partitions so concurrent
        # VSes don't evict each other's entries. 8 covers typical N:M
        # deployments with headroom; cost is a proportionally larger
        # lazy logits buffer at steady state.
        self.cache = SpeculationCache(
            max_batch_size=max_batch_size,
            num_speculative_tokens=self.K,
            fan_out=self.fan_out,
            vocab_size=self.vocab_size,
            device=self.device,
            dtype=self.dtype,
            max_verify_servers=8,
        )

        # --- Initialize OutcomePredictor ---
        # The geometric allocation (SSD paper Theorem 12) spends the
        # same total_fan_out budget non-uniformly across the K+1
        # acceptance positions, giving more candidates to the
        # positions most likely to be the actual acceptance point.
        # This raises cache hit rate without changing cache-build cost
        # (which scales with total branches, not shape).
        total_fan_out = self.fan_out * (self.K + 1)
        self.outcome_predictor = OutcomePredictor(
            num_speculative_tokens=self.K,
            total_fan_out=total_fan_out,
            acceptance_rate=0.65,
            power_law_exponent=1.5,
            device=self.device,
        )

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

        # --- Log parallel fanout status ---
        if self._use_parallel_fanout:
            logger.info(
                "Parallel fanout ENABLED: mtp_token_id=%d, "
                "all depths generated in single forward pass",
                self._mtp_token_id,
            )
    async def serve(self) -> None:
        """Main loop: accept commands from multiple verify servers.

        Single-threaded design: ZMQ I/O and GPU work run on the same
        thread.  ``_build_next_cache`` runs on a separate CUDA stream
        so it overlaps with the next ZMQ recv.
        """
        import zmq

        self._running = True
        logger.info("DraftServer serving on %s", self.bind_address)

        poller = zmq.asyncio.Poller()
        poller.register(self._socket, zmq.POLLIN)

        while self._running:
            poll_timeout_ms = 1000

            try:
                events = dict(await poller.poll(timeout=poll_timeout_ms))
            except Exception:
                if not self._running:
                    break
                logger.exception("DraftServer poll error")
                continue

            # --- Receive ZMQ messages ---
            if self._socket in events:
                msg = await self._recv_one_message(zmq)
                if msg is None:
                    if not self._running:
                        break
                    continue

                # If this is a SPECULATE and cross-VS merging is enabled,
                # try to drain additional pending SPECULATEs (NOBLOCK)
                # from DIFFERENT vs_ids, and process them as one merged
                # batch. Otherwise dispatch normally. Skip the peek when
                # only one VS is connected — the poll() call costs ~1ms
                # per round even when nothing's pending and would
                # regress single-VS latency for no benefit.
                vs_id, identity, command, frames = msg
                merged: list[tuple[str, bytes, DraftCommand,
                                   list[bytes]]] | None = None
                if (
                    command.command.upper() == "SPECULATE"
                    and len(self._verify_servers) >= 2
                ):
                    extras = await self._drain_pending_speculates(
                        zmq, already_collected={vs_id},
                    )
                    if extras:
                        merged = [msg, *extras]

                if merged is not None:
                    now = time.monotonic()
                    for item in merged:
                        self._verify_server_last_seen[item[0]] = now
                    await self._handle_speculation_merged(merged)
                else:
                    self._current_tensor_frames = frames
                    self._current_tensor_idx = 0
                    await self._dispatch(vs_id, identity, command)

            # Check for verify servers that have timed out and evict
            # their requests.
            self._check_evictions()

    async def _recv_one_message(
        self, zmq: Any,
    ) -> tuple[str, bytes, DraftCommand, list[bytes]] | None:
        """Receive one full ZMQ message; decode header. Returns None on
        recv error or malformed input. Returns (vs_id, identity, command,
        tensor_frames) on success.
        """
        try:
            frames = await self._socket.recv_multipart(
                flags=zmq.NOBLOCK
            )
        except Exception:
            if not self._running:
                return None
            logger.exception("DraftServer recv error")
            return None

        if len(frames) < 2:
            logger.warning(
                "DraftServer received malformed message with %d frames; "
                "skipping", len(frames),
            )
            return None

        identity = frames[0]
        metadata_frame = frames[1]
        tensor_frames = list(frames[2:])
        vs_id = identity.decode("utf-8", errors="replace")

        try:
            command = decode_command(metadata_frame)
        except Exception:
            logger.exception(
                "DraftServer failed to decode command from %s", vs_id,
            )
            return None

        return vs_id, identity, command, tensor_frames

    async def _drain_pending_speculates(
        self, zmq: Any, already_collected: set[str],
        peek_timeout_ms: int | None = None,
    ) -> list[tuple[str, bytes, DraftCommand, list[bytes]]]:
        """Opportunistically peek for additional pending SPECULATEs from
        DIFFERENT VSes (one peek+recv per iteration, up to one per
        connected VS minus those already collected).

        Returns a list of message tuples. Each is a SPECULATE from a
        VS not in ``already_collected``. On a non-SPECULATE message or
        a same-VS SPECULATE, the message is dispatched normally inline
        and the drain stops (no more peeks); we never drop messages.

        The first peek uses ``peek_timeout_ms`` (default
        ``self._merge_peek_timeout_ms``); subsequent peeks use 0
        (NOBLOCK) since the first delay already covered the slowest
        same-iteration arrivals.
        """
        if peek_timeout_ms is None:
            peek_timeout_ms = self._merge_peek_timeout_ms
        poller = zmq.asyncio.Poller()
        poller.register(self._socket, zmq.POLLIN)

        collected: list[tuple[str, bytes, DraftCommand, list[bytes]]] = []
        max_peeks = max(0, len(self._verify_servers) - len(already_collected))

        for i in range(max_peeks):
            timeout = peek_timeout_ms if i == 0 else 0
            try:
                events = dict(await poller.poll(timeout=timeout))
            except Exception:
                break
            if self._socket not in events:
                break

            msg = await self._recv_one_message(zmq)
            if msg is None:
                break
            vs_id2, identity2, command2, frames2 = msg
            if (
                command2.command.upper() == "SPECULATE"
                and vs_id2 not in already_collected
            ):
                collected.append(msg)
                already_collected.add(vs_id2)
                continue

            # Not a SPECULATE we can merge — dispatch inline and stop.
            self._verify_server_last_seen[vs_id2] = time.monotonic()
            self._current_tensor_frames = frames2
            self._current_tensor_idx = 0
            await self._dispatch(vs_id2, identity2, command2)
            break

        return collected

    async def shutdown(self) -> None:
        """Gracefully stop the server loop and release resources."""
        self._running = False
        if self._profiler is not None:
            self.stop_profile()
        await self._await_inflight_cache_build()
        self._cleanup()
        logger.info("DraftServer shut down.")

    # ------------------------------------------------------------------
    # Profiling (toggled via SIGUSR1/SIGUSR2 in the entrypoint)
    # ------------------------------------------------------------------

    def start_profile(self) -> None:
        """Begin a torch.profiler capture; writes on stop_profile().

        No-op if VLLM_DRAFT_TORCH_PROFILER_DIR is unset or a profile is
        already running. The captured trace covers all subsequent
        SPECULATE/PREFILL/cache-build work until ``stop_profile`` runs.
        """
        if self._profiler is not None:
            logger.info("DraftServer profile already running; ignoring start.")
            return
        prof_dir = os.environ.get("VLLM_DRAFT_TORCH_PROFILER_DIR")
        if not prof_dir:
            logger.warning(
                "DraftServer start_profile: "
                "VLLM_DRAFT_TORCH_PROFILER_DIR is not set; ignoring."
            )
            return
        os.makedirs(prof_dir, exist_ok=True)
        self._profiler_dir = prof_dir
        self._profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=True,
        )
        self._profiler.__enter__()
        logger.info("DraftServer profile started; output dir=%s", prof_dir)

    def stop_profile(self) -> None:
        """Stop the active profile capture and dump the trace."""
        if self._profiler is None:
            logger.info("DraftServer profile not running; ignoring stop.")
            return
        prof = self._profiler
        out_dir = self._profiler_dir or "/tmp"
        self._profiler = None
        self._profiler_dir = None
        prof.__exit__(None, None, None)
        out_path = os.path.join(
            out_dir,
            f"draft_server_pid{os.getpid()}_{int(time.time())}.pt.trace.json",
        )
        try:
            prof.export_chrome_trace(out_path)
            logger.info("DraftServer profile written to %s", out_path)
        except Exception:
            logger.exception(
                "DraftServer profile export failed; trace lost."
            )

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
            self.metrics.draft_speculate_total.inc()
            # Process immediately — sequential is correct for ZMQ
            # tensor transport (frames must be consumed before next msg).
            await self._handle_speculation(
                verify_server_id, identity, outcome
            )
            return

        # Non-SPECULATE commands (PREFILL / FREE_SEQ) also touch runner
        # state (KV allocation, _seq_lens, block tables). Wait for any
        # in-flight cache build to complete before executing them so
        # the two don't race.
        await self._await_inflight_cache_build()

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
            for _vs_id, ext_seq_id in keys:
                # Remap to internal seq_id
                internal_sid = self._unmap_seq_id(_vs_id, ext_seq_id)
                if internal_sid is None:
                    self._request_state.pop((_vs_id, ext_seq_id), None)
                    continue
                # Clear per-round state
                self._round_base_lens.pop(internal_sid, None)
                self._swap_states.pop(internal_sid, None)

                if runner is not None:
                    runner.free_blocks(internal_sid)

                self._request_state.pop((_vs_id, ext_seq_id), None)

            # Increment eviction counter.
            self.metrics.draft_eviction_count.inc(len(keys))

            # Release this VS's partitioned resources so its dedicated
            # blocks can be reused by peer VSes and its cache slots
            # are freed.
            if runner is not None:
                runner.recycle_dedicated_blocks(vs_id)
            if self.cache is not None:
                self.cache.reset_vs(vs_id)

            # Remove the verify server from tracking.
            self._verify_servers.pop(vs_id, None)
            self._verify_server_last_seen.pop(vs_id, None)

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

    # ------------------------------------------------------------------
    # Seq ID remapping
    # ------------------------------------------------------------------

    def _alloc_internal_seq_id(self) -> int:
        """Allocate a unique internal seq_id."""
        if self._free_internal_seq_ids:
            return self._free_internal_seq_ids.pop()
        sid = self._next_internal_seq_id
        self._next_internal_seq_id += 1
        return sid

    def _map_seq_id(self, vs_id: str, ext_seq_id: int) -> int:
        """Map (verify_server_id, external_seq_id) → internal_seq_id.

        Allocates a new internal ID on first use.
        """
        key = (vs_id, ext_seq_id)
        if key not in self._ext_to_int_seq:
            internal = self._alloc_internal_seq_id()
            self._ext_to_int_seq[key] = internal
            self._int_to_ext_seq[internal] = key
        return self._ext_to_int_seq[key]

    def _unmap_seq_id(self, vs_id: str, ext_seq_id: int) -> int | None:
        """Remove mapping and recycle the internal seq_id."""
        key = (vs_id, ext_seq_id)
        internal = self._ext_to_int_seq.pop(key, None)
        if internal is not None:
            self._int_to_ext_seq.pop(internal, None)
            self._free_internal_seq_ids.append(internal)
        return internal

    def _remap_seq_ids(
        self, vs_id: str, seq_ids: torch.Tensor
    ) -> torch.Tensor:
        """Remap a tensor of external seq_ids to internal seq_ids."""
        internal_ids = []
        for ext_id in seq_ids.tolist():
            internal_ids.append(self._map_seq_id(vs_id, int(ext_id)))
        return torch.tensor(
            internal_ids, dtype=seq_ids.dtype, device=seq_ids.device
        )

    # ------------------------------------------------------------------
    # Tensor transport helpers
    # ------------------------------------------------------------------

    def _recv_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        frames: list[bytes] | None = None,
        idx_state: list[int] | None = None,
    ) -> torch.Tensor:
        """Consume the next tensor frame from a ZMQ message.

        Default (frames=None) reads from self._current_tensor_frames /
        self._current_tensor_idx. When merging two SPECULATEs, callers
        pass explicit frames + a single-element list cursor to keep the
        two messages' tensor streams separate.
        """
        if frames is None:
            frames = self._current_tensor_frames
            idx = self._current_tensor_idx
            self._current_tensor_idx += 1
        else:
            assert idx_state is not None
            idx = idx_state[0]
            idx_state[0] += 1
        if idx >= len(frames):
            logger.warning(
                "DraftServer: tensor frame %d missing (have %d frames), "
                "returning zeros for shape=%s dtype=%s",
                idx, len(frames), shape, dtype,
            )
            return torch.zeros(shape, dtype=dtype, device=self.device)
        buf = frames[idx]
        recv_dtype = torch.float32 if dtype == torch.bfloat16 else dtype
        return torch.frombuffer(
            bytearray(buf), dtype=recv_dtype,
        ).reshape(shape).to(dtype=dtype, device=self.device)

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

    async def _await_inflight_cache_build(self) -> None:
        """Block until any scheduled cache build finishes.

        The cache build mutates the SpeculationCache and runner state
        (_seq_lens, block tables, KV cache) on the default CUDA stream.
        Any handler that also touches those must wait here first.
        """
        task = self._inflight_cache_build
        if task is None:
            return
        self._inflight_cache_build = None
        try:
            await task
        except Exception:
            logger.exception(
                "DraftServer background cache build failed"
            )

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

    def _recv_speculation_tensors(
        self,
        verify_server_id: str,
        outcome: VerificationOutcome,
        frames: list[bytes] | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None,
    ]:
        """Read seq_ids, k_accepted, bonus_tokens, temperatures off the wire.

        Order must match ZmqDraftConnector.send_and_recv_speculation.
        Remaps seq_ids into the draft-local internal numbering. If
        ``frames`` is provided, reads from that list instead of the
        instance-level cursor (used for cross-VS SPECULATE merging).
        """
        idx_state = [0] if frames is not None else None
        seq_ids = self._recv_tensor(
            outcome.seq_ids_ref.shape,
            _str_to_dtype(outcome.seq_ids_ref.dtype),
            frames=frames, idx_state=idx_state,
        )
        seq_ids = self._remap_seq_ids(verify_server_id, seq_ids)
        k_accepted = self._recv_tensor(
            outcome.k_accepted_ref.shape,
            _str_to_dtype(outcome.k_accepted_ref.dtype),
            frames=frames, idx_state=idx_state,
        )
        bonus_tokens = self._recv_tensor(
            outcome.bonus_tokens_ref.shape,
            _str_to_dtype(outcome.bonus_tokens_ref.dtype),
            frames=frames, idx_state=idx_state,
        )
        temperatures: torch.Tensor | None = None
        if outcome.temperatures_ref is not None:
            temperatures = self._recv_tensor(
                outcome.temperatures_ref.shape,
                _str_to_dtype(outcome.temperatures_ref.dtype),
                frames=frames, idx_state=idx_state,
            )
        return seq_ids, k_accepted, bonus_tokens, temperatures

    def _sync_runner_seq_lens_and_blocks(
        self,
        runner: Any,
        seq_ids_list: list[int],
        k_accepted_list: list[int],
    ) -> None:
        """Fix up per-seq KV lengths for this round and reserve headroom.

        Seq lens can be stale in two ways: the previous round ended with
        a swap (so `_seq_lens` still points at the JIT prefix), or the
        previous round's JIT extended `_seq_lens` past the accepted
        position. Both cases are reconciled from `k_accepted` relative
        to the round's base length, then we grow blocks so there is
        room for the next JIT or swap to land.
        """
        for i, sid in enumerate(seq_ids_list):
            swap_rec = self._swap_states.get(sid)
            if swap_rec is not None and getattr(
                swap_rec, "last_round_was_swap", False
            ):
                runner._seq_lens[sid] = (
                    getattr(swap_rec, "swap_prefix_len", 0)
                    + 1
                    + int(k_accepted_list[i])
                )
            elif sid in self._round_base_lens:
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

    def _apply_swap_for_hits(
        self,
        runner: Any,
        verify_server_id: str,
        seq_ids: torch.Tensor,
        seq_ids_list: list[int],
        cache_hits: torch.Tensor,
        cached_tokens: torch.Tensor,
        cached_logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> bool:
        """Swap dedicated cache blocks into the runner for hit seqs.

        Returns True if any hits were applied (so the caller knows to
        preserve swap_states); False otherwise.
        """
        hit_tables, hit_prefix_lens = self.cache.get_hit_block_tables(
            cache_hits
        )
        if hit_tables is None or hit_prefix_lens is None:
            return False

        hit_mask = cache_hits.bool()
        hit_seq_ids = seq_ids[hit_mask]
        owned, displaced = runner.swap_block_tables(
            seq_ids=hit_seq_ids,
            branch_block_tables=hit_tables,
            prefix_lens=hit_prefix_lens,
            K=self.K,
        )
        # The hit entries' dedicated blocks were reserved under THIS
        # VS — cache entries for this round's seq_ids can only come
        # from this VS's partition because internal seq_ids are
        # globally unique across VSes.
        for blocks in owned.values():
            runner.exclude_from_dedicated(blocks, verify_server_id)
        if displaced:
            runner._free_list.extend(displaced)

        hit_indices = hit_mask.nonzero(as_tuple=True)[0]
        for compact_i, idx in enumerate(hit_indices):
            sid = seq_ids_list[int(idx.item())]
            prefix_len = int(hit_prefix_lens[compact_i].item())
            runner._seq_lens[sid] = prefix_len + self.K

        draft_tokens[hit_mask] = cached_tokens[hit_mask]
        draft_logits[hit_mask] = cached_logits[hit_mask]
        return True

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

        # ---- Step 4: Apply cache hits (swap path) ----
        used_swap_for_hits = False
        if num_hits > 0 and cached_logits is not None and runner is not None:
            with torch.profiler.record_function(f"swap_hits_{num_hits}"):
                used_swap_for_hits = self._apply_swap_for_hits(
                    runner=runner,
                    verify_server_id=verify_server_id,
                    seq_ids=seq_ids,
                    seq_ids_list=seq_ids_list,
                    cache_hits=cache_hits,
                    cached_tokens=cached_tokens,
                    cached_logits=cached_logits,
                    draft_tokens=draft_tokens,
                    draft_logits=draft_logits,
                )

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

        # ---- Swap path (split owned blocks per VS) ----
        used_swap_for_hits = False
        if num_hits > 0 and cached_logits is not None:
            with torch.profiler.record_function(
                f"swap_hits_merged_{num_hits}"
            ):
                hit_tables, hit_prefix_lens = (
                    self.cache.get_hit_block_tables(cache_hits)
                )
                if hit_tables is not None and hit_prefix_lens is not None:
                    hit_seq_ids = seq_ids_cat[hit_mask]
                    owned, displaced = runner.swap_block_tables(
                        seq_ids=hit_seq_ids,
                        branch_block_tables=hit_tables,
                        prefix_lens=hit_prefix_lens,
                        K=self.K,
                    )
                    # Map each sid back to its VS so dedicated-block
                    # exclusion is scoped correctly.
                    sid_to_vs: dict[int, str] = {}
                    for i, sid in enumerate(seq_ids_list):
                        sid_to_vs[sid] = per_vs[entry_vs[i]]["vs_id"]
                    for sid, blocks in owned.items():
                        runner.exclude_from_dedicated(
                            blocks, sid_to_vs.get(sid, "__default__"),
                        )
                    if displaced:
                        runner._free_list.extend(displaced)
                    hit_indices = hit_mask.nonzero(as_tuple=True)[0]
                    for compact_i, idx in enumerate(hit_indices):
                        sid = seq_ids_list[int(idx.item())]
                        prefix_len = int(hit_prefix_lens[compact_i].item())
                        runner._seq_lens[sid] = prefix_len + self.K
                    draft_tokens[hit_mask] = cached_tokens[hit_mask]
                    draft_logits[hit_mask] = cached_logits[hit_mask]
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
                        layer_kv[:, dst_flat] = layer_kv[:, src_flat]

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

    async def _send_speculation_response(
        self,
        verify_server_id: str,
        identity: bytes,
        cache_hits: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor | None,
    ) -> None:
        """Send a SpeculationResponse back to the verify server as a
        single multipart ZMQ message: metadata + tensor frames."""
        resp = SpeculationResponse(
            cache_hits_ref=self._make_tensor_ref(cache_hits),
            draft_tokens_ref=self._make_tensor_ref(draft_tokens),
            draft_logits_ref=(
                self._make_tensor_ref(draft_logits)
                if draft_logits is not None else None
            ),
        )
        resp_bytes = encode(resp)
        tensor_frames = [
            _tensor_to_bytes(cache_hits),
            _tensor_to_bytes(draft_tokens),
        ]
        if draft_logits is not None:
            tensor_frames.append(_tensor_to_bytes(draft_logits))
        await self._socket.send_multipart(
            [identity, resp_bytes] + tensor_frames
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
                layer_kv[:, dst_flat] = layer_kv[:, src_flat]

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

    async def _handle_prefill(
        self,
        verify_server_id: str,
        identity: bytes,
        prefill: PrefillRequest,
    ) -> None:
        """Handle PREFILL: register the request, receive prompt tokens,
        and populate the draft model's KV cache."""
        key = self._register_request(verify_server_id, prefill.seq_id)
        logger.debug(
            "DraftServer PREFILL from %s, seq_id=%d, key=%s",
            verify_server_id, prefill.seq_id, key,
        )

        prompt_token_ids = self._recv_tensor(
            prefill.prompt_token_ids_ref.shape,
            _str_to_dtype(prefill.prompt_token_ids_ref.dtype),
        )

        seq_id = self._map_seq_id(verify_server_id, prefill.seq_id)
        num_tokens = torch.tensor(
            [prompt_token_ids.shape[0]],
            dtype=torch.int64, device=self.device,
        )
        seq_ids = torch.tensor(
            [seq_id], dtype=torch.int64, device=self.device,
        )

        logger.info(
            "DraftServer prefill: seq_id=%d, num_tokens=%d",
            seq_id, int(num_tokens[0].item()),
        )

        runner = self.draft_model_runner
        if runner is None or not runner._model_loaded:
            return

        try:
            runner.prefill(
                input_ids=prompt_token_ids,
                num_tokens_per_seq=num_tokens,
                seq_ids=seq_ids,
            )
        except (RuntimeError, ValueError) as e:
            logger.warning("DraftServer prefill failed: %s", e)
            return

        # Clear stale round state for freshly prefilled sequences.
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
        in ``ZmqDraftConnector.send_free_seq``), then for each
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
        seq_ids = self._recv_tensor(
            free_req.seq_ids_ref.shape,
            _str_to_dtype(free_req.seq_ids_ref.dtype),
        )

        # ---- Step 2: Free resources for each sequence ----
        runner = self.draft_model_runner
        freed = 0
        freed_internal_sids: list[int] = []
        for ext_sid in seq_ids.tolist():
            ext_sid = int(ext_sid)
            # Remap to internal seq_id and release the mapping
            sid = self._unmap_seq_id(verify_server_id, ext_sid)
            if sid is None:
                # Unknown seq — skip
                continue

            freed_internal_sids.append(sid)

            # Clear per-round state
            self._round_base_lens.pop(sid, None)
            self._swap_states.pop(sid, None)

            if runner is not None:
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

        # Drop ONLY this VS's cache entries for the freed sids. The
        # freed sids might be recycled by the shared
        # ``_free_internal_seq_ids`` pool and reassigned to another VS,
        # so leaving their cache entries in place would risk a peer-VS
        # SPECULATE hitting stale pre-free data. But other sequences
        # belonging to this same VS are still live — wiping the whole
        # partition (as we used to) collapsed cache hit rate to ~50%
        # under 2V+1D at high concurrency. Surgical removal preserves
        # entries for active sequences while still protecting against
        # stale-sid cross-VS collisions.
        if self.cache is not None and freed_internal_sids:
            self.cache.drop_entries_by_seq_ids(
                verify_server_id, freed_internal_sids
            )

    async def _handle_exit(
        self, verify_server_id: str, identity: bytes
    ) -> None:
        """Handle EXIT command from a verify server.

        Cleans up all state for the disconnecting verify server,
        including its SpeculationCache partition and dedicated-block
        pool (so those resources can be reused by other VSes).
        """
        keys = list(self._verify_servers.get(verify_server_id, set()))
        for key in keys:
            self._request_state.pop(key, None)
        self._verify_servers.pop(verify_server_id, None)

        runner = self.draft_model_runner
        if runner is not None:
            runner.recycle_dedicated_blocks(verify_server_id)
        if self.cache is not None:
            self.cache.reset_vs(verify_server_id)

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
