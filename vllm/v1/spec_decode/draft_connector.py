# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport between verify servers and draft servers (N:M).

Defines the ``DraftConnector`` ABC and its concrete ZMQ-only
implementation ``ZmqDraftConnector``. All messages are sent as a
single ZMQ multipart frame where the first frame is msgpack-encoded
metadata and the remaining frames are raw tensor bytes.

Wire layout:

    Verify → Draft (SPECULATE / PREFILL / FREE_SEQ):
        [identity, metadata_bytes, tensor_0_bytes, tensor_1_bytes, ...]

    Draft → Verify (SpeculationResponse):
        [identity, metadata_bytes, tensor_0_bytes, tensor_1_bytes, ...]

The DEALER socket on the verify side sets its identity; the ROUTER on
the draft side prepends it on receive.

The tensors in SPECULATE at steady state are tiny (~32-bytes each:
seq_ids, k_accepted, bonus_tokens). At default F=1 the response is
similarly small. Round-trip latency on localhost is ~1 ms.
"""

from __future__ import annotations

import ctypes
import itertools
import logging
import mmap
import os
import pickle
import socket
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    import torch.multiprocessing.reductions  # noqa: F401

logger = logging.getLogger(__name__)


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "bool": torch.bool,
}


def _dtype_to_str(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


def _str_to_dtype(s: str) -> torch.dtype:
    return _DTYPE_MAP[s]


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    """Serialize a tensor to bytes, casting bfloat16→float32 for numpy."""
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float32)
    return t.numpy().tobytes()


def _bytes_to_tensor(
    buf: bytes,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Deserialize bytes to a tensor, handling bfloat16 via float32."""
    recv_dtype = torch.float32 if dtype == torch.bfloat16 else dtype
    t = torch.frombuffer(bytearray(buf), dtype=recv_dtype).reshape(shape)
    return t.to(dtype=dtype, device=device)


@dataclass
class PendingSpeculation:
    """Opaque handle returned by dispatch_speculation, consumed by
    await_speculation. Contents are connector-specific."""

    connector: "DraftConnector"
    state: Any  # connector-specific payload


class DraftConnector(ABC):
    """Transport interface to a single draft server."""

    @abstractmethod
    async def send_and_recv_speculation(
        self,
        batch_size: int,
        seq_ids: torch.Tensor,
        k_accepted: torch.Tensor,
        bonus_tokens: torch.Tensor,
        temperatures: torch.Tensor | None,
        needs_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        ...

    def dispatch_speculation(
        self,
        batch_size: int,
        seq_ids: torch.Tensor,
        k_accepted: torch.Tensor,
        bonus_tokens: torch.Tensor,
        temperatures: torch.Tensor | None,
        needs_logits: bool = False,
    ) -> PendingSpeculation:
        """Fire a SPECULATE without waiting for the reply.

        Default implementation stores the arguments and defers the whole
        round-trip until ``await_speculation`` — behaves identically to
        ``send_and_recv_speculation`` for transports that cannot overlap
        (e.g. ZMQ). Fast transports (CUDA IPC) override this to fire the
        send eagerly so the drafter's compute overlaps with the caller's
        subsequent CPU work.
        """
        return PendingSpeculation(
            connector=self,
            state={
                "batch_size": batch_size,
                "seq_ids": seq_ids,
                "k_accepted": k_accepted,
                "bonus_tokens": bonus_tokens,
                "temperatures": temperatures,
                "needs_logits": needs_logits,
            },
        )

    async def await_speculation(
        self, handle: PendingSpeculation,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Wait for the reply to a previously-dispatched SPECULATE.

        Default implementation performs the deferred round-trip. Fast
        transports override with a poll on their transport-specific
        completion signal.
        """
        s = handle.state
        return await self.send_and_recv_speculation(
            batch_size=s["batch_size"],
            seq_ids=s["seq_ids"],
            k_accepted=s["k_accepted"],
            bonus_tokens=s["bonus_tokens"],
            temperatures=s["temperatures"],
            needs_logits=s["needs_logits"],
        )

    @abstractmethod
    async def send_prefill(
        self,
        seq_id: int,
        prompt_token_ids: torch.Tensor,
    ) -> None:
        ...

    @abstractmethod
    async def send_free_seq(self, seq_ids: torch.Tensor) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class ZmqDraftConnector(DraftConnector):
    """Pure-ZMQ transport to a draft server.

    All tensors for a given command are sent as additional frames in
    the same multipart message as the command metadata. That keeps
    metadata and tensors always in sync — avoids frame-desync races
    that arise with split transports.
    """

    def __init__(
        self,
        address: str,
        verify_server_id: str,
        device: torch.device,
        timeout_ms: int = 5000,
    ) -> None:
        import zmq
        import zmq.asyncio

        self._address = address
        self._verify_server_id = verify_server_id
        self._device = device
        self._timeout_ms = timeout_ms

        self._buffer_counter = itertools.count()
        self._connected = False
        self._zmq_ctx: zmq.asyncio.Context | None = None
        self._socket: zmq.asyncio.Socket | None = None

        self._connect()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        import zmq

        from vllm.utils.network_utils import make_zmq_socket

        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass

        if self._zmq_ctx is None:
            import zmq.asyncio
            self._zmq_ctx = zmq.asyncio.Context()

        self._socket = make_zmq_socket(
            self._zmq_ctx,
            self._address,
            zmq.DEALER,
            bind=False,
            identity=self._verify_server_id.encode("utf-8"),
            linger=1000,
        )

        if self._timeout_ms > 0:
            self._socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
            self._socket.setsockopt(zmq.SNDTIMEO, self._timeout_ms)

        self._connected = True
        logger.info(
            "ZmqDraftConnector connected to %s as %s",
            self._address,
            self._verify_server_id,
        )

    def _reconnect(self) -> None:
        logger.warning(
            "ZmqDraftConnector reconnecting to %s", self._address
        )
        self._connected = False
        try:
            self._connect()
        except Exception:
            logger.exception(
                "ZmqDraftConnector failed to reconnect to %s",
                self._address,
            )

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # ZMQ multipart helpers
    # ------------------------------------------------------------------

    async def _send_multipart(
        self, metadata: bytes, tensors: list[torch.Tensor]
    ) -> None:
        import zmq

        assert self._socket is not None
        frames: list[bytes] = [metadata]
        for t in tensors:
            frames.append(_tensor_to_bytes(t))
        try:
            await self._socket.send_multipart(frames)
        except zmq.Again:
            self._connected = False
            raise ConnectionError(
                f"ZMQ send timeout to {self._address}"
            )
        except zmq.ZMQError as exc:
            self._connected = False
            raise ConnectionError(
                f"ZMQ send error to {self._address}: {exc}"
            ) from exc

    async def _recv_multipart(self) -> tuple[bytes, list[bytes]]:
        """Receive a multipart ZMQ message → (metadata_bytes, tensor_frames)."""
        import zmq

        assert self._socket is not None
        try:
            frames = await self._socket.recv_multipart()
        except zmq.Again:
            self._connected = False
            raise ConnectionError(
                f"ZMQ recv timeout from {self._address}"
            )
        except zmq.ZMQError as exc:
            self._connected = False
            raise ConnectionError(
                f"ZMQ recv error from {self._address}: {exc}"
            ) from exc
        if not frames:
            raise ConnectionError("Empty ZMQ response")
        return frames[0], frames[1:]

    def _make_tensor_ref(self, tensor: torch.Tensor) -> "TensorRef":
        from vllm.v1.spec_decode.draft_data_models import TensorRef

        return TensorRef(
            shape=tuple(tensor.shape),
            dtype=_dtype_to_str(tensor.dtype),
            buffer_id=str(next(self._buffer_counter)),
            nbytes=tensor.nelement() * tensor.element_size(),
        )

    def _ensure_connected(self) -> None:
        if not self._connected:
            self._reconnect()
            if not self._connected:
                raise ConnectionError(
                    f"Not connected to draft server at {self._address}"
                )

    # ------------------------------------------------------------------
    # DraftConnector interface
    # ------------------------------------------------------------------

    async def send_and_recv_speculation(
        self,
        batch_size: int,
        seq_ids: torch.Tensor,
        k_accepted: torch.Tensor,
        bonus_tokens: torch.Tensor,
        temperatures: torch.Tensor | None,
        needs_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        from torch.profiler import record_function
        from vllm.v1.spec_decode.draft_data_models import (
            SpeculationResponse,
            VerificationOutcome,
            decode,
            encode_command,
        )

        self._ensure_connected()

        with record_function("connector: prep_outcome"):
            _seq_ids = seq_ids[:batch_size].to(torch.int64).reshape(-1)
            _k_accepted = k_accepted[:batch_size].to(torch.int64).reshape(-1)
            _bonus_tokens = bonus_tokens[:batch_size].to(torch.int64).reshape(-1)

            tensor_list: list[torch.Tensor] = [
                _seq_ids, _k_accepted, _bonus_tokens,
            ]

            temps_ref = None
            if temperatures is not None:
                _temps = temperatures[:batch_size].to(torch.float32)
                temps_ref = self._make_tensor_ref(_temps)
                tensor_list.append(_temps)

            outcome = VerificationOutcome(
                verify_server_id=self._verify_server_id,
                batch_size=batch_size,
                seq_ids_ref=self._make_tensor_ref(_seq_ids),
                k_accepted_ref=self._make_tensor_ref(_k_accepted),
                bonus_tokens_ref=self._make_tensor_ref(_bonus_tokens),
                temperatures_ref=temps_ref,
                needs_logits=needs_logits,
            )
            cmd_bytes = encode_command("SPECULATE", outcome)

        with record_function("connector: zmq_send_multipart"):
            await self._send_multipart(cmd_bytes, tensor_list)

        with record_function("connector: zmq_recv_multipart"):
            resp_bytes, tensor_frames = await self._recv_multipart()

        resp = decode(resp_bytes, SpeculationResponse)

        cache_hits = _bytes_to_tensor(
            tensor_frames[0],
            resp.cache_hits_ref.shape,
            _str_to_dtype(resp.cache_hits_ref.dtype),
            self._device,
        )
        draft_tokens = _bytes_to_tensor(
            tensor_frames[1],
            resp.draft_tokens_ref.shape,
            _str_to_dtype(resp.draft_tokens_ref.dtype),
            self._device,
        )
        draft_logits: torch.Tensor | None = None
        if resp.draft_logits_ref is not None:
            draft_logits = _bytes_to_tensor(
                tensor_frames[2],
                resp.draft_logits_ref.shape,
                _str_to_dtype(resp.draft_logits_ref.dtype),
                self._device,
            )
        return cache_hits, draft_tokens, draft_logits

    async def send_prefill(
        self,
        seq_id: int,
        prompt_token_ids: torch.Tensor,
    ) -> None:
        from vllm.v1.spec_decode.draft_data_models import (
            PrefillRequest,
            encode_command,
        )

        self._ensure_connected()

        _prompt = prompt_token_ids.to(torch.int64)
        prefill = PrefillRequest(
            verify_server_id=self._verify_server_id,
            seq_id=seq_id,
            prompt_token_ids_ref=self._make_tensor_ref(_prompt),
        )
        cmd_bytes = encode_command("PREFILL", prefill)
        await self._send_multipart(cmd_bytes, [_prompt])

    async def send_free_seq(self, seq_ids: torch.Tensor) -> None:
        from vllm.v1.spec_decode.draft_data_models import (
            FreeSeqRequest,
            encode_command,
        )

        self._ensure_connected()

        _seq_ids = seq_ids.to(torch.int64)
        free_req = FreeSeqRequest(
            verify_server_id=self._verify_server_id,
            seq_ids_ref=self._make_tensor_ref(_seq_ids),
        )
        cmd_bytes = encode_command("FREE_SEQ", free_req)
        await self._send_multipart(cmd_bytes, [_seq_ids])

    def close(self) -> None:
        self._connected = False
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None
        if self._zmq_ctx is not None:
            try:
                self._zmq_ctx.term()
            except Exception:
                pass
            self._zmq_ctx = None
        logger.info(
            "ZmqDraftConnector closed for %s", self._verify_server_id
        )


# ---------------------------------------------------------------------------
# CUDA-IPC connector: SPECULATE via cuda IPC, everything else via inherited ZMQ
# ---------------------------------------------------------------------------

IPC_N_SLOTS = 16
IPC_DEFAULT_MAX_BATCH = 128
IPC_DEFAULT_K = 8


class _IpcSharedBuffer:
    """Ring of ``N_SLOTS`` slots on the verifier GPU, shared with drafter
    via ``cudaIpcGetMemHandle``.

    Buffer layout is fixed at handshake time using ``max_batch`` and ``K``
    upper bounds. Each slot holds one SPECULATE request + response payload.

    GPU-side doorbells live alongside the payload tensors so both sides
    can write them via kernel-queued ops on the default stream. This
    keeps request writes ordered before the doorbell without requiring
    a CPU sync (the write is scheduled after the copies on the same
    stream, so the drafter observes them in order via P2P).
    """

    def __init__(self, device: torch.device, max_batch: int, K: int):
        self.device = device
        self.max_batch = max_batch
        self.K = K
        B = max_batch
        with torch.cuda.device(device):
            self.req_seq_ids = torch.zeros(
                IPC_N_SLOTS, B, dtype=torch.int64, device=device,
            )
            self.req_k_accepted = torch.zeros(
                IPC_N_SLOTS, B, dtype=torch.int64, device=device,
            )
            self.req_bonus_tokens = torch.zeros(
                IPC_N_SLOTS, B, dtype=torch.int64, device=device,
            )
            self.req_temperatures = torch.ones(
                IPC_N_SLOTS, B, dtype=torch.float32, device=device,
            )
            self.resp_cache_hits = torch.zeros(
                IPC_N_SLOTS, B, dtype=torch.int64, device=device,
            )
            self.resp_draft_tokens = torch.zeros(
                IPC_N_SLOTS, B, K, dtype=torch.int64, device=device,
            )
            # GPU-side doorbells. int32 per slot per direction.
            self.dbell_req_gpu = torch.zeros(
                IPC_N_SLOTS, dtype=torch.int32, device=device,
            )
            self.dbell_resp_gpu = torch.zeros(
                IPC_N_SLOTS, dtype=torch.int32, device=device,
            )

    _TENSOR_NAMES = (
        "req_seq_ids", "req_k_accepted", "req_bonus_tokens",
        "req_temperatures", "resp_cache_hits", "resp_draft_tokens",
        "dbell_req_gpu", "dbell_resp_gpu",
    )

    def as_handle_dict(self) -> dict:
        """Serialize each tensor as a ``(rebuild_fn, args)`` tuple. Uses
        ``torch.multiprocessing.reductions.reduce_tensor`` which calls
        ``cudaIpcGetMemHandle`` internally."""
        import torch.multiprocessing.reductions as tmr
        return {n: tmr.reduce_tensor(getattr(self, n)) for n in self._TENSOR_NAMES}

    @classmethod
    def from_handle_dict(
        cls, hdict: dict, max_batch: int, K: int,
    ) -> "_IpcSharedBuffer":
        """Reconstruct on the drafter side. Each rebuild call ends up in
        ``cudaIpcOpenMemHandle``."""
        obj = cls.__new__(cls)
        for n, (rebuild, args) in hdict.items():
            setattr(obj, n, rebuild(*args))
        obj.device = obj.req_seq_ids.device
        obj.max_batch = max_batch
        obj.K = K
        return obj


class _IpcDoorbells:
    """Mmap-backed doorbell pair. int32 per slot per direction (req/resp).

    Reader/writer identify each other via a monotonic seq_no; the seq_no
    is encoded together with ``batch_size`` in the upper bits so the
    drafter can shape its response reads correctly without a separate
    metadata channel.

    Layout: ``[req_0, req_1, ..., req_N-1, resp_0, resp_1, ..., resp_N-1]``
    """

    _NBYTES = 8 * IPC_N_SLOTS  # 4 bytes req + 4 bytes resp per slot

    def __init__(self, shm_path: str, create: bool):
        self.shm_path = shm_path
        if create:
            with open(shm_path, "wb") as f:
                f.write(b"\0" * self._NBYTES)
        fd = os.open(shm_path, os.O_RDWR)
        self._mm = mmap.mmap(
            fd, self._NBYTES, mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE,
        )
        os.close(fd)
        base = ctypes.addressof(ctypes.c_char.from_buffer(self._mm))
        self._req = (ctypes.c_int32 * IPC_N_SLOTS).from_address(base)
        self._resp = (ctypes.c_int32 * IPC_N_SLOTS).from_address(
            base + 4 * IPC_N_SLOTS,
        )

    def set_req(self, slot: int, value: int) -> None:
        self._req[slot] = value

    def get_req(self, slot: int) -> int:
        return self._req[slot]

    def set_resp(self, slot: int, value: int) -> None:
        self._resp[slot] = value

    def get_resp(self, slot: int) -> int:
        return self._resp[slot]

    def close(self) -> None:
        try:
            self._mm.close()
        except Exception:
            pass


class CudaIpcDraftConnector(ZmqDraftConnector):
    """SPECULATE via CUDA IPC, PREFILL/FREE_SEQ inherited from ZMQ.

    The IPC handshake is layered on top of the ZMQ control channel that
    the base class already establishes. At construction time we allocate
    a shared ring buffer on the verifier GPU, expose it via ``cudaIpcGet``,
    and ship handles + a doorbell shm path to the draft server through a
    new ``IPC_HANDSHAKE`` msgpack command.

    Falls back to the base-class ZMQ path if the handshake fails, IPC
    isn't supported by the peer, or ``force_zmq`` is set.
    """

    def __init__(
        self,
        address: str,
        verify_server_id: str,
        device: torch.device,
        max_batch: int = IPC_DEFAULT_MAX_BATCH,
        K: int = IPC_DEFAULT_K,
        timeout_ms: int = 5000,
        force_zmq: bool = False,
    ) -> None:
        super().__init__(address, verify_server_id, device, timeout_ms)
        self._ipc_max_batch = max_batch
        self._ipc_K = K
        self._ipc_ready = False
        self._buf: _IpcSharedBuffer | None = None
        self._dbell: _IpcDoorbells | None = None
        self._shm_path: str | None = None
        self._next_slot = 0
        self._slot_seqs = [0] * IPC_N_SLOTS

        self._force_zmq = force_zmq
        if force_zmq:
            logger.info(
                "CudaIpcDraftConnector: force_zmq=True, running as plain ZMQ",
            )

        # Handshake is deferred. Owners must call ``establish_ipc()`` (or
        # ``await async_establish_ipc()``) exactly once on the event loop
        # that will service SPECULATEs. Doing it here in ``__init__`` is
        # unsafe because no loop is bound to the current thread yet.

    async def async_establish_ipc(self) -> None:
        """Perform the IPC handshake against the drafter, awaiting on the
        current event loop. Idempotent: no-op if already ready or
        force_zmq is set."""
        if self._force_zmq or self._ipc_ready:
            return
        try:
            await self._establish_ipc_async()
            self._ipc_ready = True
            logger.info(
                "CudaIpcDraftConnector: IPC handshake succeeded with %s",
                self._address,
            )
        except Exception:
            logger.exception(
                "CudaIpcDraftConnector: IPC handshake failed; falling back "
                "to ZMQ transport for SPECULATE",
            )
            self._teardown_ipc()

    def establish_ipc(self) -> None:
        """Synchronous version — creates a temporary event loop for the
        handshake. Convenience for callers that don't own a loop yet."""
        import asyncio
        if self._force_zmq or self._ipc_ready:
            return
        # We can't just call ``asyncio.get_event_loop`` here because the
        # base class's ZMQ socket is already bound to a specific asyncio
        # context (its ZMQ context). Re-use whatever loop the base uses.
        # Safest: use asyncio.new_event_loop() and set it as current,
        # then reset on exit.
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.async_establish_ipc())
        finally:
            asyncio.set_event_loop(None)
            loop.close()

    async def _establish_ipc_async(self) -> None:
        """Allocate GPU buffer + doorbells, ship handles to drafter via
        an IPC_HANDSHAKE msgpack command, wait for ACK."""
        from vllm.v1.spec_decode.draft_data_models import (
            IpcHandshake,
            IpcHandshakeAck,
            decode,
            encode_command,
        )

        self._buf = _IpcSharedBuffer(
            self._device, self._ipc_max_batch, self._ipc_K,
        )
        # GPU-side doorbells live inside self._buf. The shm_path field
        # in the handshake is retained for wire-format compatibility
        # (drafter accepts an empty string).
        self._shm_path = ""
        self._dbell = None

        gpu_handles = pickle.dumps(self._buf.as_handle_dict())
        handshake = IpcHandshake(
            verify_server_id=self._verify_server_id,
            shm_path=self._shm_path,
            max_batch=self._ipc_max_batch,
            K=self._ipc_K,
            n_slots=IPC_N_SLOTS,
            gpu_handles_pickle=gpu_handles,
        )

        cmd_bytes = encode_command("IPC_HANDSHAKE", handshake)
        await self._send_multipart(cmd_bytes, [])
        resp_bytes, _frames = await self._recv_multipart()
        ack = decode(resp_bytes, IpcHandshakeAck)
        if not ack.ok:
            raise ConnectionError(
                f"Drafter rejected IPC_HANDSHAKE: {ack.error}",
            )

    def _teardown_ipc(self) -> None:
        self._dbell = None
        self._shm_path = None
        self._buf = None
        self._ipc_ready = False

    # ------------------------------------------------------------------
    # Split dispatch/await path
    # ------------------------------------------------------------------

    def dispatch_speculation(
        self,
        batch_size: int,
        seq_ids: torch.Tensor,
        k_accepted: torch.Tensor,
        bonus_tokens: torch.Tensor,
        temperatures: torch.Tensor | None,
        needs_logits: bool = False,
    ) -> PendingSpeculation:
        if not self._ipc_ready or needs_logits:
            # Fall back to the base default (deferred ZMQ round-trip)
            return super().dispatch_speculation(
                batch_size, seq_ids, k_accepted, bonus_tokens,
                temperatures, needs_logits,
            )
        if batch_size > self._ipc_max_batch:
            logger.warning(
                "batch_size %d > ipc_max_batch %d; using ZMQ fallback",
                batch_size, self._ipc_max_batch,
            )
            return super().dispatch_speculation(
                batch_size, seq_ids, k_accepted, bonus_tokens,
                temperatures, needs_logits,
            )

        slot = self._next_slot
        self._next_slot = (self._next_slot + 1) % IPC_N_SLOTS
        seq_no = self._slot_seqs[slot] + 1
        self._slot_seqs[slot] = seq_no

        buf = self._buf
        assert buf is not None
        buf.req_seq_ids[slot, :batch_size].copy_(
            seq_ids[:batch_size], non_blocking=True,
        )
        buf.req_k_accepted[slot, :batch_size].copy_(
            k_accepted[:batch_size], non_blocking=True,
        )
        buf.req_bonus_tokens[slot, :batch_size].copy_(
            bonus_tokens[:batch_size], non_blocking=True,
        )
        if temperatures is not None:
            buf.req_temperatures[slot, :batch_size].copy_(
                temperatures[:batch_size], non_blocking=True,
            )
        else:
            buf.req_temperatures[slot, :batch_size].fill_(1.0)

        assert batch_size < (1 << 16), "batch_size doesn't fit in 16 bits"
        # Wrap seq_no to 16 bits so the encoding stays in int32. Consumers
        # compare against their own remembered seq_no, so wrap-around is
        # safe as long as inflight-depth stays under 2^16.
        seq16 = seq_no & 0xFFFF
        if seq16 == 0:
            # Skip zero — used as "unset" sentinel by the drafter poll
            self._slot_seqs[slot] = 1
            seq16 = 1
        encoded = (batch_size << 16) | seq16

        # Fire the doorbell via a kernel-queued fill_ on the default
        # stream. Because it's queued AFTER the payload copies above,
        # the drafter's P2P read of dbell_req_gpu will only observe
        # the new value once the copies have committed. No CPU sync
        # required — the verifier CPU thread returns immediately.
        buf.dbell_req_gpu[slot].fill_(encoded)

        return PendingSpeculation(
            connector=self,
            state={"slot": slot, "seq16": seq16, "batch_size": batch_size},
        )

    async def await_speculation(
        self, handle: PendingSpeculation,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not self._ipc_ready or handle.state.get("seq16") is None:
            return await super().await_speculation(handle)

        import asyncio

        s = handle.state
        slot = s["slot"]
        seq16 = s["seq16"]
        batch_size = s["batch_size"]

        buf = self._buf
        assert buf is not None

        # Poll GPU-side response doorbell via .item() — issues a D2H
        # each call (~10 µs). Cheaper than a full stream sync. Async
        # yields let other coroutines run between polls.
        deadline = asyncio.get_event_loop().time() + self._timeout_ms / 1000.0
        dbell_slice = buf.dbell_resp_gpu[slot:slot + 1]
        while int(dbell_slice.item()) != seq16:
            if asyncio.get_event_loop().time() > deadline:
                self._connected = False
                raise ConnectionError(
                    f"IPC recv timeout on slot {slot} seq {seq16}",
                )
            await asyncio.sleep(0)

        cache_hits = (
            buf.resp_cache_hits[slot, :batch_size].clone().to(torch.bool)
        )
        draft_tokens = buf.resp_draft_tokens[slot, :batch_size].clone()
        return cache_hits, draft_tokens, None

    async def send_and_recv_speculation(
        self,
        batch_size: int,
        seq_ids: torch.Tensor,
        k_accepted: torch.Tensor,
        bonus_tokens: torch.Tensor,
        temperatures: torch.Tensor | None,
        needs_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Blocking API: implemented via dispatch + await so callers that
        haven't been split-aware still work."""
        if not self._ipc_ready or needs_logits:
            return await super().send_and_recv_speculation(
                batch_size, seq_ids, k_accepted, bonus_tokens,
                temperatures, needs_logits,
            )
        handle = self.dispatch_speculation(
            batch_size, seq_ids, k_accepted, bonus_tokens,
            temperatures, needs_logits,
        )
        return await self.await_speculation(handle)

    def close(self) -> None:
        self._teardown_ipc()
        super().close()


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------


def validate_draft_server_connectivity(
    draft_addresses: list[str],
    timeout_ms: int = 5000,
) -> None:
    """Check that at least one draft server is reachable at startup."""
    if not draft_addresses:
        raise RuntimeError(
            "disagg_draft_addresses is empty — at least one draft "
            "server address must be provided."
        )

    reachable: list[str] = []
    unreachable: list[str] = []

    for addr in draft_addresses:
        try:
            import socket
            from urllib.parse import urlparse

            parsed = urlparse(addr)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port
            if port is None:
                raise ValueError(f"No port in address: {addr}")

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout_ms / 1000.0)
            sock.connect((host, port))
            sock.close()
            logger.info(
                "Draft server %s is reachable (TCP connect OK).", addr
            )
            reachable.append(addr)
        except Exception as exc:
            logger.warning(
                "Draft server %s not reachable: %s", addr, exc
            )
            unreachable.append(addr)

    if not reachable:
        raise RuntimeError(
            f"No draft servers reachable at startup. "
            f"Tried {len(draft_addresses)} address(es): "
            f"{', '.join(draft_addresses)}. "
            f"Ensure draft servers are running and accessible."
        )
