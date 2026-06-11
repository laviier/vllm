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
    _str_to_dtype,
)
from vllm.v1.spec_decode.draft_data_models import (
    DraftCommand,
    FreeSeqRequest,
    PrefillRequest,
    VerificationOutcome,
    decode,
    decode_command,
    encode,  # used by HEALTHCHECK_ACK reply
)
from vllm.v1.spec_decode.draft_server_mixins import (
    DraftServerCacheBuildMixin,
    DraftServerFanoutMixin,
    DraftServerSeqIdMixin,
    DraftServerSpeculateMixin,
    DraftServerTransportMixin,
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


class DraftServer(
    DraftServerSeqIdMixin,
    DraftServerTransportMixin,
    DraftServerFanoutMixin,
    DraftServerCacheBuildMixin,
    DraftServerSpeculateMixin,
):
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
        self._mtp_token_id: int | None = spec_config.disagg_mtp_token_id
        if self._use_parallel_fanout and self._mtp_token_id is None:
            raise ValueError(
                "disagg_parallel_fanout=True requires disagg_mtp_token_id "
                "to be set (the trained mask token's vocab id). For a "
                "checkpoint produced via BISGlora's extend_vocab, this is "
                "typically the highest valid vocab id (e.g. 128256 for "
                "Llama-3 + one appended mask token)."
            )

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
