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
    IpcHandshake,
    IpcHandshakeAck,
    PrefillRequest,
    SpeculationResponse,
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

        # Internal accumulators for rolling cache hit rate. Hits live
        # on the GPU as a 0-d tensor so SPECULATE can ``+= sum()`` the
        # cache_hits mask without a sync; we materialize the running
        # total only when the gauge is refreshed (every
        # ``_hit_rate_sync_period`` rounds).
        self._total_lookups: int = 0
        self._total_hits: int = 0
        self._pending_hits_gpu: torch.Tensor | None = None
        self._hit_rate_sync_round: int = 0
        self._hit_rate_sync_period: int = 64

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


class _DraftServerIpcPeer:
    """Per-VS state for the CUDA-IPC SPECULATE transport.

    Attached to a verify server via IPC_HANDSHAKE. Holds the IPC-opened
    GPU ring buffer (owned by the verifier) and per-slot last-seen
    sequence numbers so we don't service the same request twice.

    Doorbells are GPU tensors inside ``gpu_bufs`` (``dbell_req_gpu`` /
    ``dbell_resp_gpu``), read/written via ``.item()`` and ``.fill_()``
    on the verifier's GPU. shm_path is retained for backward compat
    (unused when GPU-doorbells are present).
    """

    def __init__(
        self,
        verify_server_id: str,
        shm_path: str,
        max_batch: int,
        K: int,
        n_slots: int,
        gpu_bufs: dict[str, torch.Tensor],
    ) -> None:
        self.verify_server_id = verify_server_id
        self.shm_path = shm_path
        self.max_batch = max_batch
        self.K = K
        self.n_slots = n_slots
        self.gpu_bufs = gpu_bufs
        # `resp_draft_tokens.device` is the verifier's GPU (tensors were
        # allocated on that GPU by the verifier and IPC-opened here).
        self.target_device: torch.device = gpu_bufs[
            "resp_draft_tokens"
        ].device
        self.last_seen = [0] * n_slots
        # GPU doorbells (IPC-shared via cudaIpc).
        self._dbell_req = gpu_bufs["dbell_req_gpu"]
        self._dbell_resp = gpu_bufs["dbell_resp_gpu"]
        # CPU-side pinned staging for one-shot D2H of all doorbell
        # values. Cheap to reuse; a single D2H picks up all slots at
        # once so the per-tick poll cost is one memcpy, not n_slots.
        self._dbell_req_cpu = torch.zeros(
            n_slots, dtype=torch.int32, pin_memory=True,
        )
        # Pinned staging for seq_ids D2H (side-stream) and remapped-ids
        # H2D (side-stream). Both must be pinned so the non-blocking
        # copies don't serialize with the default stream (cache_build
        # kernels). Sized to max_batch — the actual per-round B <= this.
        self._seq_ids_cpu = torch.zeros(
            max_batch, dtype=torch.int64, pin_memory=True,
        )
        self._remap_ids_cpu = torch.zeros(
            max_batch, dtype=torch.int64, pin_memory=True,
        )
        # Same pattern for k_accepted D2H — used to build the CPU list
        # that ``_sync_runner_seq_lens_and_blocks`` needs. Skipping
        # this side-stream fetch causes the inner handler's
        # ``k_accepted.tolist()`` to stall ~3 ms behind cache_build.
        self._k_accepted_cpu = torch.zeros(
            max_batch, dtype=torch.int64, pin_memory=True,
        )
        # Dedicated side stream for the poll D2H and seq_ids H2D/D2H.
        # Using the default stream serializes with cache_build kernels;
        # a side stream lets these fire while cache_build is running.
        with torch.cuda.device(self.target_device):
            self._poll_stream = torch.cuda.Stream()

    def poll_all_reqs(self) -> torch.Tensor:
        """Read the full request-doorbell vector in one D2H on a side
        stream. Returns a CPU int32 tensor of length ``n_slots``.

        This D2H does NOT wait for prior default-stream work (e.g.
        an in-flight cache_build). The read reflects whatever value
        the doorbell tensor holds at the time the copy is scheduled;
        because doorbell writes on the verifier are ordered-after
        their payload copies on the verifier's default stream, a
        published new value implies the payload is fully written.
        """
        with torch.cuda.stream(self._poll_stream):
            self._dbell_req_cpu.copy_(self._dbell_req, non_blocking=True)
        self._poll_stream.synchronize()
        return self._dbell_req_cpu

    def set_resp(self, slot: int, value: int) -> None:
        """Kernel-queued write of the response doorbell. Ordering vs.
        prior response-tensor copies is preserved by the default stream."""
        self._dbell_resp[slot].fill_(value)

    def close(self) -> None:
        pass


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

        # Hit-row swap deferred from the SPECULATE handler to the start
        # of the next cache_build. Set by _handle_speculation_inner /
        # _handle_speculation_merged_inner, consumed in the prologue of
        # _run_cache_build / _run_cache_build_merged.
        self._pending_swap: dict[str, Any] | None = None
        self._pending_swap_merged: dict[str, Any] | None = None

        # Glue logits computed by the fused cleanup+glue forward in
        # cache_build's prologue, consumed by ``_build_next_cache`` /
        # ``_run_cache_build_merged`` in place of running ``glue_decode``
        # again. Shape [B_total, V] aligned with this round's seq_ids.
        # Single-VS and merged-VS paths are mutually exclusive per
        # round (one of ``_handle_speculation`` /
        # ``_handle_speculation_merged`` runs, never both), so the
        # producer/consumer pair is always a single round of the same
        # path.
        self._pending_glue_logits: torch.Tensor | None = None

        # Last speculation seq_ids — stored by _handle_speculation_inner,
        # consumed by _handle_speculation for post-response cache building.
        self._last_spec_seq_ids: torch.Tensor | None = None
        # CPU-side mirror of _last_spec_seq_ids, materialized once in
        # the SPECULATE handler so downstream cache_build consumers can
        # reuse it instead of paying another GPU→host sync.
        self._last_spec_seq_ids_cpu: list[int] | None = None

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

        # Messages peeked during cross-VS merge drain that we couldn't
        # safely dispatch inline (because doing so would violate per-VS
        # FIFO with the held primary SPECULATE). The serve loop drains
        # this queue before polling ZMQ.
        self._deferred_messages: list[
            tuple[str, bytes, DraftCommand, list[bytes]]
        ] = []

        # ----- Per-VS CUDA-IPC state (Path C SPECULATE transport) -----
        # If a verify server sent an IPC_HANDSHAKE, we hold its shared
        # ring buffer + doorbells here. Otherwise it stays on ZMQ and
        # this dict is empty. Poll path in serve() scans this dict each
        # tick alongside the ZMQ poll.
        # Keyed by verify_server_id.
        self._ipc_peers: dict[str, "_DraftServerIpcPeer"] = {}
        # Round-robin cursor for ``_poll_ipc_speculates`` to prevent
        # a fast peer from starving a slower peer under multi-VS load.
        self._ipc_poll_cursor: int = 0

        # Pre-allocated pinned CPU staging for cache_build's per-round
        # H2Ds. ``torch.tensor(python_list, device=cuda)`` in
        # _allocate_branch_blocks_and_copy_kv was ~2.5 ms per call
        # because pageable H2D serializes with cache_build's own
        # default-stream kernels. Pinned copies do not.
        # Sizes:
        #  base_lens_pin: [max_batch_size] — one entry per seq_id.
        #  ded_blocks_pin: [MAX_BRANCHES] — one entry per branch block.
        self._cb_base_lens_pin = torch.zeros(
            max_batch_size, dtype=torch.int64, pin_memory=True,
        )
        self._cb_base_lens_gpu = torch.zeros(
            max_batch_size, dtype=torch.int64, device=self.device,
        )
        self._cb_ded_blocks_pin = torch.zeros(
            self.MAX_BRANCHES, dtype=torch.int64, pin_memory=True,
        )
        self._cb_ded_blocks_gpu = torch.zeros(
            self.MAX_BRANCHES, dtype=torch.int64, device=self.device,
        )

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
            # Fast path: check IPC doorbells first — service any pending
            # SPECULATEs before falling through to ZMQ. Only costs a few
            # int32 loads per registered peer when idle.
            if self._ipc_peers:
                pending_ipc = self._poll_ipc_all_pending()
                # Optional accumulation window: if only a subset of
                # peers are ready, briefly yield and re-poll so late-
                # arriving doorbells can join this merged round.
                # Default is 0 (no wait): every micro-second of wait
                # adds to the ITL of the peer that already arrived,
                # and empirically that regresses more than the extra
                # merge coverage helps at 2V/3V c=8. Set
                # ``VLLM_DISAGG_IPC_ACCUM_US=N`` (e.g. 150) to enable
                # a wait when higher merge coverage would be worth
                # more than sub-N-µs response latency for a peer that
                # arrived first.
                if pending_ipc and len(pending_ipc) < len(self._ipc_peers):
                    accum_us = int(
                        os.environ.get("VLLM_DISAGG_IPC_ACCUM_US", "0")
                    )
                    if accum_us > 0:
                        await asyncio.sleep(accum_us / 1_000_000.0)
                        # Re-poll to pick up any straggler doorbells that
                        # fired during the sleep. Existing entries in
                        # ``pending_ipc`` were marked ``last_seen``, so
                        # this call returns ONLY NEW pending items.
                        extras = self._poll_ipc_all_pending()
                        if extras:
                            pending_ipc.extend(extras)
                if pending_ipc:
                    # Drain ALL pending ZMQ commands before running the
                    # SPECULATE(s). PREFILL/FREE_SEQ from any VS must
                    # apply before we look up in the cache, since the
                    # verifier sends them over ZMQ *before* the matching
                    # SPECULATE over IPC. IPC arrives instantly while
                    # ZMQ has wire latency, so without this drain the
                    # SPECULATE can leapfrog its own PREFILL/FREE_SEQ
                    # and corrupt cache state under sustained load.
                    #
                    # Dispatching peer-VS commands here (rather than
                    # deferring to ``_deferred_messages``) prevents
                    # starvation under multi-VS IPC load — the serve
                    # loop otherwise ``continue``s past the deferred-
                    # queue drain on every hot tick.
                    while True:
                        try:
                            zmq_frames = await self._socket.recv_multipart(
                                flags=zmq.NOBLOCK,
                            )
                        except zmq.Again:
                            break
                        except Exception:
                            if not self._running:
                                break
                            logger.exception(
                                "DraftServer ZMQ drain error"
                            )
                            break
                        if len(zmq_frames) < 2:
                            continue
                        identity = zmq_frames[0]
                        metadata_frame = zmq_frames[1]
                        tensor_frames = list(zmq_frames[2:])
                        vs_id_zmq = identity.decode(
                            "utf-8", errors="replace",
                        )
                        try:
                            command = decode_command(metadata_frame)
                        except Exception:
                            logger.exception(
                                "DraftServer failed to decode command from %s",
                                vs_id_zmq,
                            )
                            continue
                        self._current_tensor_frames = tensor_frames
                        self._current_tensor_idx = 0
                        await self._dispatch(vs_id_zmq, identity, command)
                    # Drain any older deferred messages skipped by prior
                    # IPC-fast-path ``continue``s (legacy safety net).
                    while self._deferred_messages:
                        msg = self._deferred_messages.pop(0)
                        vs_id_msg, identity_msg, command_msg, frames_msg = msg
                        self._current_tensor_frames = frames_msg
                        self._current_tensor_idx = 0
                        await self._dispatch(
                            vs_id_msg, identity_msg, command_msg,
                        )
                    # Await any in-flight cache build (SPECULATE is
                    # single-threaded with cache_build).
                    await self._await_inflight_cache_build()
                    # Late-arrival re-poll: the ``_await_inflight_...``
                    # above can take 10-20 ms under heavy load. During
                    # that wait, additional peers' doorbells can fire.
                    # Re-poll so those get folded into this merged
                    # round instead of triggering a separate solo
                    # round on the next serve iteration (which would
                    # cost another ~13 ms of drafter time).
                    if len(pending_ipc) < len(self._ipc_peers):
                        late = self._poll_ipc_all_pending()
                        if late:
                            pending_ipc.extend(late)
                    # After the ZMQ drain, a VS may have EXITed and
                    # had its cache/state torn down via ``reset_vs``.
                    # Its IPC peer is still registered (EXIT only wipes
                    # cache/dedicated blocks, not the IPC ring), and
                    # its stashed doorbell tick would try to lookup
                    # against entries that no longer exist. Skip such
                    # VSes here so we don't scramble the merged
                    # cache.lookup.
                    live_ipc = [
                        item for item in pending_ipc
                        if item[0] in self._verify_servers
                    ]
                    if not live_ipc:
                        self._check_evictions()
                        continue
                    # Multi-VS: merge into ONE drafter forward when we
                    # have multiple pending IPC SPECULATEs. Fall back to
                    # single-VS for len==1.
                    if len(live_ipc) == 1:
                        vs_id, slot, batch_size, seq16 = live_ipc[0]
                        await self._handle_ipc_speculation(
                            vs_id, slot, batch_size, seq16,
                        )
                    else:
                        await self._handle_ipc_speculation_merged(
                            live_ipc,
                        )
                    self._check_evictions()
                    continue

            # Drain any messages that the merge peek deferred to keep
            # per-VS FIFO ordering. These take priority over fresh ZMQ
            # recv so a deferred FREE_SEQ never overtakes the SPECULATE
            # that was held alongside it.
            if self._deferred_messages:
                msg = self._deferred_messages.pop(0)
            else:
                # Use a short poll timeout when IPC peers are active so
                # we return to poll doorbells quickly; longer timeout
                # when only ZMQ is in play (idle-friendly).
                poll_timeout_ms = 1 if self._ipc_peers else 1000
                try:
                    events = dict(await poller.poll(timeout=poll_timeout_ms))
                except Exception:
                    if not self._running:
                        break
                    logger.exception("DraftServer poll error")
                    continue
                if self._socket not in events:
                    self._check_evictions()
                    continue
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
        VS not in ``already_collected``. Anything else — non-SPECULATE
        commands and SPECULATEs from a VS already collected — is
        deferred to ``self._deferred_messages`` so the next serve-loop
        iteration handles it AFTER the held primary, preserving
        per-VS FIFO ordering. We never drop messages.

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
            vs_id2, _identity2, command2, _frames2 = msg
            if (
                command2.command.upper() == "SPECULATE"
                and vs_id2 not in already_collected
            ):
                collected.append(msg)
                already_collected.add(vs_id2)
                continue

            # Can't merge: either non-SPECULATE or a SPECULATE from a
            # VS already collected. Inline dispatch would invert the
            # per-VS FIFO with the held primary (e.g., a FREE_SEQ
            # could free seq_ids the held SPECULATE references).
            # Defer to the serve loop's deferred queue and stop.
            self._deferred_messages.append(msg)
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

        elif cmd == "IPC_HANDSHAKE":
            await self._handle_ipc_handshake(
                verify_server_id, identity, command,
            )

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

            # Unregister the request. _register_request was called in
            # _handle_prefill with the external seq_id; mirror that here.
            self._unregister_request(verify_server_id, ext_sid)

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

    async def _handle_ipc_handshake(
        self,
        verify_server_id: str,
        identity: bytes,
        command: DraftCommand,
    ) -> None:
        """Handle IPC_HANDSHAKE: open the verifier-shared GPU ring buffer
        + doorbells and register the peer.

        Once registered, subsequent SPECULATEs from this VS travel via the
        IPC ring (polled in ``serve``); the base-class ZMQ path is used
        only for PREFILL/FREE_SEQ/HEALTHCHECK/EXIT and for SPECULATE
        fallback if the peer registration or handshake fails.

        Replies with a raw msgpack-encoded ``IpcHandshakeAck`` (NOT wrapped
        in ``DraftCommand``) — the connector's handshake code decodes it
        directly.
        """
        import pickle

        try:
            handshake = decode(command.payload, IpcHandshake)
            gpu_handles = pickle.loads(handshake.gpu_handles_pickle)
            gpu_bufs: dict[str, torch.Tensor] = {}
            for name, (rebuild, args) in gpu_handles.items():
                gpu_bufs[name] = rebuild(*args)

            peer = _DraftServerIpcPeer(
                verify_server_id=verify_server_id,
                shm_path=handshake.shm_path,
                max_batch=handshake.max_batch,
                K=handshake.K,
                n_slots=handshake.n_slots,
                gpu_bufs=gpu_bufs,
            )
            # If a previous IPC session existed for this VS (e.g. after a
            # reconnect), close the stale peer first.
            existing = self._ipc_peers.pop(verify_server_id, None)
            if existing is not None:
                existing.close()
            self._ipc_peers[verify_server_id] = peer
            logger.info(
                "DraftServer registered IPC peer %s: max_batch=%d K=%d "
                "n_slots=%d target_device=%s",
                verify_server_id, handshake.max_batch, handshake.K,
                handshake.n_slots, peer.target_device,
            )
            ack = IpcHandshakeAck(ok=True)
        except Exception as e:
            logger.exception(
                "DraftServer IPC_HANDSHAKE failed for %s", verify_server_id,
            )
            ack = IpcHandshakeAck(ok=False, error=repr(e))

        try:
            await self._socket.send_multipart([identity, encode(ack)])
        except Exception:
            logger.exception(
                "DraftServer failed to send IPC_HANDSHAKE ack to %s",
                verify_server_id,
            )

    # ------------------------------------------------------------------
    # IPC SPECULATE poll path
    # ------------------------------------------------------------------

    def _poll_ipc_all_pending(
        self,
    ) -> list[tuple[str, int, int, int]]:
        """Return one pending SPECULATE (if any) per peer, in
        round-robin order. Enables cross-VS merged dispatch.

        Rotates the start of the sweep via ``_ipc_poll_cursor`` so no
        peer gets starved when its immediate neighbour is always hot.
        """
        vs_ids = list(self._ipc_peers.keys())
        if not vs_ids:
            return []
        n = len(vs_ids)
        start = self._ipc_poll_cursor % n
        pending: list[tuple[str, int, int, int]] = []
        for i in range(n):
            vs_id = vs_ids[(start + i) % n]
            peer = self._ipc_peers[vs_id]
            dbell_cpu = peer.poll_all_reqs()
            values = dbell_cpu.tolist()
            # Only pick ONE pending slot per peer per call — matching
            # the verifier's dispatch/await protocol, only one
            # SPECULATE is inflight per VS at any time.
            for slot in range(peer.n_slots):
                encoded = values[slot]
                if encoded == peer.last_seen[slot] or encoded == 0:
                    continue
                if encoded < 0:
                    logger.info(
                        "IPC peer %s signalled shutdown on slot %d",
                        vs_id, slot,
                    )
                    peer.last_seen[slot] = encoded
                    continue
                peer.last_seen[slot] = encoded
                batch_size = (encoded >> 16) & 0xFFFF
                seq16 = encoded & 0xFFFF
                pending.append((vs_id, slot, batch_size, seq16))
                break
        # Advance cursor by 1 so the next round starts on a fresh
        # peer even if only some peers had pending work.
        self._ipc_poll_cursor = (start + 1) % n
        return pending

    def _poll_ipc_speculates(
        self,
    ) -> tuple[str, int, int, int] | None:
        """Scan all IPC peers for a pending SPECULATE. Returns
        ``(verify_server_id, slot, batch_size, seq16)`` for the first
        pending request found, or ``None`` if none is ready.

        Called from ``serve()`` each tick. Each peer does one D2H of
        the whole doorbell vector, then compares against ``last_seen``
        entirely on CPU.

        Poll order rotates across peers so no VS gets starved: dict
        iteration is insertion-ordered, and always picking the first
        pending under sustained multi-VS load would starve the 3rd+ VS
        (observed as 600-1200 ms SPECULATE latency).
        """
        vs_ids = list(self._ipc_peers.keys())
        if not vs_ids:
            return None
        n = len(vs_ids)
        # ``_ipc_poll_cursor`` advances each time we successfully
        # service a peer, so the NEXT tick starts at the peer AFTER
        # the one we just served — classic round-robin.
        start = self._ipc_poll_cursor % n
        for i in range(n):
            vs_id = vs_ids[(start + i) % n]
            peer = self._ipc_peers[vs_id]
            dbell_cpu = peer.poll_all_reqs()
            values = dbell_cpu.tolist()
            for slot in range(peer.n_slots):
                encoded = values[slot]
                if encoded == peer.last_seen[slot] or encoded == 0:
                    continue
                if encoded < 0:
                    logger.info(
                        "IPC peer %s signalled shutdown on slot %d",
                        vs_id, slot,
                    )
                    peer.last_seen[slot] = encoded
                    continue
                peer.last_seen[slot] = encoded
                batch_size = (encoded >> 16) & 0xFFFF
                seq16 = encoded & 0xFFFF
                self._ipc_poll_cursor = (start + i + 1) % n
                return (vs_id, slot, batch_size, seq16)
        return None

    async def _handle_ipc_speculation(
        self,
        verify_server_id: str,
        slot: int,
        batch_size: int,
        seq16: int,
    ) -> None:
        """Service a SPECULATE that arrived via the IPC ring.

        Loads tensors out of the peer's shared ring, dispatches to the
        existing ``_handle_speculation_inner`` (same code path as ZMQ,
        via the ``preloaded_tensors`` shortcut), copies the response
        tensors into the ring, and bumps the response doorbell.
        Cache_build is scheduled the same way as the ZMQ path.
        """
        peer = self._ipc_peers.get(verify_server_id)
        if peer is None:
            return

        # Track VS activity so the eviction timer doesn't fire.
        self._verify_server_last_seen[verify_server_id] = time.monotonic()

        buf = peer.gpu_bufs
        B = batch_size
        try:
            # Fast-path seq_ids remap using the side stream in both
            # directions:
            #  1. D2H: read raw ring seq_ids into pinned CPU buffer
            #     (side stream — doesn't wait for cache_build kernels
            #     on the default stream).
            #  2. CPU remap: ext_id → internal_id via mapping dict.
            #  3. H2D: pinned CPU → GPU staging → slice (side stream —
            #     same non-blocking benefit).
            # Do both D2Hs (seq_ids, k_accepted) on the side stream at
            # once so we only synchronize once. Both feed the inner
            # handler as preloaded CPU lists to skip its own
            # ``.tolist()`` calls.
            seq_ids_ring = buf["req_seq_ids"][slot, :B]
            k_accepted_ring = buf["req_k_accepted"][slot, :B]
            seq_ids_cpu = peer._seq_ids_cpu[:B]
            k_accepted_cpu = peer._k_accepted_cpu[:B]
            with torch.cuda.stream(peer._poll_stream):
                seq_ids_cpu.copy_(seq_ids_ring, non_blocking=True)
                k_accepted_cpu.copy_(k_accepted_ring, non_blocking=True)
            peer._poll_stream.synchronize()
            # Remap on CPU. Keep the Python lists around too so the
            # inner handler can skip its own ``.tolist()`` calls.
            ext_list = seq_ids_cpu.tolist()
            k_accepted_list = k_accepted_cpu.tolist()
            internal_ids: list[int] = []
            remap_view = peer._remap_ids_cpu[:B]
            for i, ext in enumerate(ext_list):
                mapped = self._map_seq_id(verify_server_id, int(ext))
                internal_ids.append(mapped)
                remap_view[i] = mapped
            # Allocate + write on the DEFAULT stream so the caching
            # allocator tags the tensor's owning stream to match its
            # downstream consumers (cache.lookup runs on the default
            # stream). A previous version wrote this on
            # ``peer._poll_stream`` while allocating on the default
            # stream; that stream/allocator mismatch surfaced under
            # multi-VS merged load as a spurious device-side assert
            # (masked when CUDA_LAUNCH_BLOCKING=1). Pinned source keeps
            # the copy non-blocking on the default stream.
            seq_ids = torch.empty(
                B, dtype=torch.int64, device=self.device,
            )
            seq_ids.copy_(remap_view, non_blocking=True)

            k_accepted = buf["req_k_accepted"][slot, :B].to(
                device=self.device, non_blocking=True, copy=True,
            )
            bonus_tokens = buf["req_bonus_tokens"][slot, :B].to(
                device=self.device, non_blocking=True, copy=True,
            )
            temperatures = buf["req_temperatures"][slot, :B].to(
                device=self.device, non_blocking=True, copy=True,
            )

            # Synthesize a VerificationOutcome for the inner handler.
            # The refs are unused when preloaded_tensors is provided.
            outcome = VerificationOutcome(
                verify_server_id=verify_server_id,
                batch_size=B,
                seq_ids_ref=self._make_tensor_ref(seq_ids),
                k_accepted_ref=self._make_tensor_ref(k_accepted),
                bonus_tokens_ref=self._make_tensor_ref(bonus_tokens),
                temperatures_ref=self._make_tensor_ref(temperatures),
                needs_logits=False,
            )

            # Reuse the ZMQ inner handler wholesale — same cache lookup,
            # miss-fill, pending_swap plumbing, _last_* stashing, etc.
            # ``ipc_send_ctx`` makes the inner handler write the IPC
            # response tensors AND fire the doorbell as soon as it has
            # cache_hits + draft_tokens (i.e. right after cache.lookup),
            # so the verifier unblocks ~10 ms sooner than under the old
            # "compute everything, then send" flow. The remaining
            # per-round staging (draft_logits blend, pending_swap
            # setup, _last_* clones) happens after the send and
            # overlaps with the verifier's target forward.
            result = await self._handle_speculation_inner(
                verify_server_id,
                identity=b"",  # unused on IPC path
                outcome=outcome,
                preloaded_tensors=(
                    seq_ids, k_accepted, bonus_tokens, temperatures,
                ),
                preloaded_seq_ids_list=internal_ids,
                preloaded_k_accepted_list=k_accepted_list,
                ipc_send_ctx=(peer, slot, seq16),
            )
            if result is not None:
                # Defensive fallback: inner-handler-sends path returns
                # None on IPC. Reaching here means the caller's
                # ipc_send_ctx wasn't honored (should not happen).
                cache_hits, draft_tokens, _dl, _nl = result
                buf["resp_cache_hits"][slot, :B].copy_(
                    cache_hits.to(torch.int64), non_blocking=True,
                )
                buf["resp_draft_tokens"][slot, :B].copy_(
                    draft_tokens, non_blocking=True,
                )
                peer.set_resp(slot, seq16)

            # Cache_build is scheduled inside the inner
            # (``_phase_b_and_cache_build_solo``) so the serve loop
            # returns to polling doorbells right after Phase A.

            self.metrics.draft_speculate_total.inc()

        except Exception:
            logger.exception(
                "DraftServer IPC speculation failed for %s slot %d",
                verify_server_id, slot,
            )
            try:
                buf["resp_cache_hits"][slot, :B].zero_()
                buf["resp_draft_tokens"][slot, :B].zero_()
                peer.set_resp(slot, seq16)
            except Exception:
                logger.exception(
                    "DraftServer IPC fallback response failed",
                )

    async def _handle_ipc_speculation_merged(
        self,
        pending: list[tuple[str, int, int, int]],
    ) -> None:
        """Cross-VS merged variant of ``_handle_ipc_speculation``.

        Loads tensors from EACH peer's ring on their shared side stream,
        remaps per-VS seq_ids to internal ids, and dispatches to
        ``_handle_speculation_merged_inner`` with the preloaded per-VS
        payload. The inner concatenates along the batch dim and runs a
        single merged cache_lookup + response fill, then writes
        responses back into EACH peer's ring via the IPC branch of the
        inner handler's send loop.
        """
        if not pending:
            return
        # Track VS activity so eviction timers don't fire.
        now = time.monotonic()
        for vs_id, _, _, _ in pending:
            self._verify_server_last_seen[vs_id] = now

        # ---- Phase A: pipelined D2H across all peers, ONE host sync ----
        # Fire the seq_ids/k_accepted D2H on each peer's poll stream in
        # a single loop so all D2Hs run concurrently, then synchronize
        # each stream once. Previously each peer had its own
        # ``synchronize()`` inside the read loop, so N peers cost N
        # host barriers — visible under 2-VS/3-VS merge.
        prefetch: list[tuple[str, int, int, int, Any]] = []
        for vs_id, slot, B, seq16 in pending:
            peer = self._ipc_peers.get(vs_id)
            if peer is None:
                continue
            buf = peer.gpu_bufs
            seq_ids_ring = buf["req_seq_ids"][slot, :B]
            k_accepted_ring = buf["req_k_accepted"][slot, :B]
            seq_ids_cpu = peer._seq_ids_cpu[:B]
            k_accepted_cpu = peer._k_accepted_cpu[:B]
            with torch.cuda.stream(peer._poll_stream):
                seq_ids_cpu.copy_(seq_ids_ring, non_blocking=True)
                k_accepted_cpu.copy_(k_accepted_ring, non_blocking=True)
            prefetch.append((vs_id, slot, B, seq16, peer))
        # Single-pass sync — poll streams have already been kicked off
        # in parallel above.
        for _vs_id, _slot, _B, _seq16, peer in prefetch:
            peer._poll_stream.synchronize()

        per_vs: list[dict[str, Any]] = []
        try:
            for vs_id, slot, B, seq16, peer in prefetch:
                buf = peer.gpu_bufs
                seq_ids_cpu = peer._seq_ids_cpu[:B]
                k_accepted_cpu = peer._k_accepted_cpu[:B]
                ext_list = seq_ids_cpu.tolist()
                k_accepted_list = k_accepted_cpu.tolist()
                internal_ids: list[int] = []
                remap_view = peer._remap_ids_cpu[:B]
                for i, ext in enumerate(ext_list):
                    mapped = self._map_seq_id(vs_id, int(ext))
                    internal_ids.append(mapped)
                    remap_view[i] = mapped
                # All GPU tensors below are allocated AND written on the
                # drafter's default stream so the caching allocator tags
                # them correctly for downstream default-stream consumers
                # (torch.cat + cache.lookup). A previous version wrote
                # them on ``peer._poll_stream`` and hit a stream/allocator
                # race that surfaced as a spurious device-side assert
                # under sustained multi-VS load (masked when
                # CUDA_LAUNCH_BLOCKING=1).
                seq_ids = torch.empty(
                    B, dtype=torch.int64, device=self.device,
                )
                seq_ids.copy_(remap_view, non_blocking=True)
                k_accepted = buf["req_k_accepted"][slot, :B].to(
                    device=self.device, non_blocking=True, copy=True,
                )
                bonus_tokens = buf["req_bonus_tokens"][slot, :B].to(
                    device=self.device, non_blocking=True, copy=True,
                )
                temperatures = buf["req_temperatures"][slot, :B].to(
                    device=self.device, non_blocking=True, copy=True,
                )
                outcome = VerificationOutcome(
                    verify_server_id=vs_id,
                    batch_size=B,
                    seq_ids_ref=self._make_tensor_ref(seq_ids),
                    k_accepted_ref=self._make_tensor_ref(k_accepted),
                    bonus_tokens_ref=self._make_tensor_ref(bonus_tokens),
                    temperatures_ref=self._make_tensor_ref(temperatures),
                    needs_logits=False,
                )
                per_vs.append({
                    "vs_id": vs_id,
                    "identity": b"",  # unused on IPC path
                    "outcome": outcome,
                    "B": B,
                    "seq_ids": seq_ids,
                    "k_accepted": k_accepted,
                    "bonus_tokens": bonus_tokens,
                    "temperatures": temperatures,
                    "seq_ids_list": internal_ids,
                    "k_accepted_list": k_accepted_list,
                    # IPC response-fill hook consumed by
                    # ``_handle_speculation_merged_inner``.
                    "ipc_peer": peer,
                    "ipc_slot": slot,
                    "ipc_seq16": seq16,
                })

            if not per_vs:
                return

            # ``items`` is only used by the ZMQ recv path when
            # ``preloaded_per_vs`` is None. Pass a shape-only stub.
            items_stub = [
                (p["vs_id"], b"", None, [])  # type: ignore[list-item]
                for p in per_vs
            ]
            await self._handle_speculation_merged_inner(
                items_stub, preloaded_per_vs=per_vs,
            )

        except Exception:
            logger.exception(
                "DraftServer IPC merged speculation failed for %s",
                [p[0] for p in pending],
            )
            # Fallback: write zeros to each peer's ring so verifiers
            # don't time out.
            for vs_id, slot, B, seq16 in pending:
                peer = self._ipc_peers.get(vs_id)
                if peer is None:
                    continue
                try:
                    buf = peer.gpu_bufs
                    buf["resp_cache_hits"][slot, :B].zero_()
                    buf["resp_draft_tokens"][slot, :B].zero_()
                    peer.set_resp(slot, seq16)
                except Exception:
                    logger.exception(
                        "DraftServer IPC merged fallback failed for %s",
                        vs_id,
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
        for peer in self._ipc_peers.values():
            peer.close()
        self._ipc_peers.clear()
        self._request_state.clear()
        self._verify_servers.clear()
        self._verify_server_last_seen.clear()
