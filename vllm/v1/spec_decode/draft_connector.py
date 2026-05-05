# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Abstract transport interface for disaggregated speculative decoding (N:M).

Defines the ``DraftConnector`` ABC that replaces the NCCL-based
``DisaggDraftCommunicator`` with a network-capable transport supporting
ZMQ metadata + NCCL/TCP tensor payloads.  Concrete implementations
(``ZmqNcclDraftConnector``, ``ZmqTcpDraftConnector``) provide the
actual transport logic.

The connector supports the same command set as the existing protocol:
SPECULATE, PREFILL, FREE_SEQ, EXIT.

Protocol (ZMQ fallback, no NCCL):
  All messages are sent as a SINGLE ZMQ multipart message to avoid
  frame desync between metadata and tensor frames.

  Verify→Draft (SPECULATE/PREFILL/FREE_SEQ):
    [metadata_bytes, tensor_0_bytes, tensor_1_bytes, ...]

  Draft→Verify (SpeculationResponse):
    [metadata_bytes, tensor_0_bytes, tensor_1_bytes, ...]

  The DEALER socket identity is set as the socket identity, so the
  ROUTER on the draft server side sees:
    [identity, b"", metadata_bytes, tensor_0_bytes, ...]
"""

from __future__ import annotations

import itertools
import logging
from abc import ABC, abstractmethod

import torch

logger = logging.getLogger(__name__)


class DraftConnector(ABC):
    """Abstract transport replacing DisaggDraftCommunicator."""

    async def send_verification_outcome(
        self,
        batch_size: int,
        seq_ids: torch.Tensor,
        k_accepted: torch.Tensor,
        bonus_tokens: torch.Tensor,
        temperatures: torch.Tensor | None,
        hidden_states: torch.Tensor | None,
        aux_hidden_states: torch.Tensor | None,
        extend_counts: torch.Tensor | None,
        extend_hidden_states: torch.Tensor | None,
        extend_token_ids: torch.Tensor | None,
        needs_logits: bool = False,
    ) -> None: ...

    @abstractmethod
    async def send_and_recv_speculation(
        self,
        batch_size: int,
        seq_ids: torch.Tensor,
        k_accepted: torch.Tensor,
        bonus_tokens: torch.Tensor,
        temperatures: torch.Tensor | None,
        hidden_states: torch.Tensor | None,
        aux_hidden_states: torch.Tensor | None,
        extend_counts: torch.Tensor | None,
        extend_hidden_states: torch.Tensor | None,
        extend_token_ids: torch.Tensor | None,
        needs_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]: ...

    @abstractmethod
    async def send_prefill(
        self,
        seq_id: int,
        prompt_token_ids: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> None: ...

    @abstractmethod
    async def send_free_seq(self, seq_ids: torch.Tensor) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Serialize a tensor to bytes, casting bfloat16→float32 for numpy compat."""
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


class ZmqNcclDraftConnector(DraftConnector):
    """ZMQ for metadata + NCCL for tensor payloads (or ZMQ fallback).

    When ``process_group`` is ``None``, all tensors are sent inline as
    additional frames in the same ZMQ multipart message as the metadata.
    This avoids the frame-desync race condition that occurs when metadata
    and tensors are sent as separate messages.
    """

    def __init__(
        self,
        address: str,
        verify_server_id: str,
        process_group: "torch.distributed.ProcessGroup | None",
        peer_rank: int,
        device: torch.device,
        timeout_ms: int = 5000,
    ) -> None:
        import zmq
        import zmq.asyncio

        self._address = address
        self._verify_server_id = verify_server_id
        self._pg = process_group
        self._peer_rank = peer_rank
        self._device = device
        self._timeout_ms = timeout_ms

        self._buffer_counter = itertools.count()
        self._connected = False
        self._zmq_ctx: zmq.asyncio.Context | None = None
        self._socket: zmq.asyncio.Socket | None = None

        self._connect()

        # ZMQ-only tensor transport — NCCL handshake disabled.
        # For standalone drafters the tensors are tiny (~100 bytes),
        # so ZMQ serialization overhead is negligible (~0.4ms).
        # NCCL would require all processes to see all GPUs (no
        # CUDA_VISIBLE_DEVICES isolation), which complicates deployment.
        # The NCCL code paths remain in _nccl_send/_nccl_recv for
        # future use with EAGLE hidden states if needed.

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        import zmq
        import zmq.asyncio

        from vllm.utils.network_utils import make_zmq_socket

        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass

        if self._zmq_ctx is None:
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
            "ZmqNcclDraftConnector connected to %s as %s",
            self._address,
            self._verify_server_id,
        )

    def _reconnect(self) -> None:
        logger.warning("ZmqNcclDraftConnector reconnecting to %s", self._address)
        self._connected = False
        try:
            self._connect()
        except Exception:
            logger.exception(
                "ZmqNcclDraftConnector failed to reconnect to %s", self._address
            )

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # NCCL handshake
    # ------------------------------------------------------------------

    def _nccl_handshake(self) -> None:
        """Establish an NCCL PG with the draft server via ZMQ handshake.

        The verify server hosts a TCPStore (master=True) on a random
        port, sends the port to the draft server via a HANDSHAKE ZMQ
        message, and waits for the draft server to connect.
        """
        import asyncio
        from datetime import timedelta

        import torch.distributed as dist

        from vllm.utils.network_utils import get_open_port

        store_port = get_open_port()
        store_host = "0.0.0.0"

        logger.info(
            "ZmqNcclDraftConnector: initiating NCCL handshake with %s "
            "(store port %d)",
            self._address,
            store_port,
        )

        # Send HANDSHAKE over ZMQ
        from vllm.v1.spec_decode.draft_data_models import (
            HandshakeRequest,
            HandshakeResponse,
            decode,
            encode_command,
        )

        hs_req = HandshakeRequest(
            verify_server_id=self._verify_server_id,
            nccl_store_host=store_host,
            nccl_store_port=store_port,
        )
        cmd_bytes = encode_command("HANDSHAKE", hs_req)

        # Send handshake and receive response synchronously
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._send_multipart(cmd_bytes, []))
        finally:
            loop.close()

        # Start TCPStore as master — draft server will connect to this
        store = dist.TCPStore(
            host_name=store_host,
            port=store_port,
            world_size=2,
            is_master=True,
            timeout=timedelta(seconds=30),
        )

        # Create NCCL PG: verify=rank 0, draft=rank 1
        pg = dist.ProcessGroupNCCL(
            store,
            rank=0,
            size=2,
            timeout=timedelta(hours=24),
        )

        self._pg = pg
        self._peer_rank = 1

        # Warm up the NCCL communicator with a timeout.
        # The first send/recv triggers lazy NCCL init (a collective).
        # If GPUs can't see each other, this hangs — so we use a thread
        # with a timeout to detect and abort.
        import threading

        warmup = torch.zeros(1, dtype=torch.int64, device=self._device)
        warmup_ok = [False]

        def _do_warmup():
            try:
                pg.send([warmup], 1, 0).wait()
                pg.recv([warmup], 1, 0).wait()
                warmup_ok[0] = True
            except Exception:
                pass

        t = threading.Thread(target=_do_warmup, daemon=True)
        t.start()
        t.join(timeout=10)  # 10 second timeout

        if not warmup_ok[0]:
            self._pg = None
            self._peer_rank = 0
            raise RuntimeError(
                "NCCL warmup timed out — GPUs may not have P2P visibility. "
                "Use --draft-server-device instead of CUDA_VISIBLE_DEVICES."
            )

        # Wait for handshake response
        loop = asyncio.new_event_loop()
        try:
            resp_bytes, _ = loop.run_until_complete(self._recv_multipart())
        finally:
            loop.close()

        resp = decode(resp_bytes, HandshakeResponse)
        if not resp.success:
            self._pg = None
            raise RuntimeError(
                f"NCCL handshake failed: {resp.error}"
            )

        logger.info(
            "ZmqNcclDraftConnector: NCCL PG established with %s "
            "(verify=rank0, draft=rank1)",
            self._address,
        )

    # ------------------------------------------------------------------
    # NCCL helpers (used when process_group is set)
    # ------------------------------------------------------------------

    def _nccl_send(self, tensor: torch.Tensor) -> None:
        assert self._pg is not None
        self._pg.send(
            [tensor.contiguous().to(self._device)], self._peer_rank, 0
        ).wait()

    def _nccl_recv(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        assert self._pg is not None
        tensor = torch.empty(shape, dtype=dtype, device=self._device)
        self._pg.recv([tensor], self._peer_rank, 0).wait()
        return tensor

    # ------------------------------------------------------------------
    # ZMQ multipart helpers (used when process_group is None)
    # ------------------------------------------------------------------

    async def _send_multipart(self, metadata: bytes, tensors: list[torch.Tensor]) -> None:
        """Send metadata + tensors as a single ZMQ multipart message."""
        import zmq

        assert self._socket is not None
        frames: list[bytes] = [metadata]
        for t in tensors:
            frames.append(_tensor_to_bytes(t))
        try:
            await self._socket.send_multipart(frames)
        except zmq.Again:
            self._connected = False
            raise ConnectionError(f"ZMQ send timeout to {self._address}")
        except zmq.ZMQError as exc:
            self._connected = False
            raise ConnectionError(f"ZMQ send error to {self._address}: {exc}") from exc

    async def _recv_multipart(self) -> tuple[bytes, list[bytes]]:
        """Receive a single ZMQ multipart message, returning (metadata, tensor_frames).

        The ROUTER sends [identity, metadata, t0, t1, ...].
        The DEALER receives [metadata, t0, t1, ...] (identity stripped by ROUTER).
        """
        import zmq

        assert self._socket is not None
        try:
            frames = await self._socket.recv_multipart()
        except zmq.Again:
            self._connected = False
            raise ConnectionError(f"ZMQ recv timeout from {self._address}")
        except zmq.ZMQError as exc:
            self._connected = False
            raise ConnectionError(f"ZMQ recv error from {self._address}: {exc}") from exc

        if not frames:
            raise ConnectionError("Empty ZMQ response")

        metadata = frames[0]
        tensor_frames = frames[1:]
        return metadata, tensor_frames

    # ------------------------------------------------------------------
    # TensorRef helpers
    # ------------------------------------------------------------------

    def _make_tensor_ref(self, tensor: torch.Tensor) -> "TensorRef":
        from vllm.v1.spec_decode.draft_data_models import TensorRef

        return TensorRef(
            shape=tuple(tensor.shape),
            dtype=_dtype_to_str(tensor.dtype),
            buffer_id=str(next(self._buffer_counter)),
            nbytes=tensor.nelement() * tensor.element_size(),
        )

    # ------------------------------------------------------------------
    # DraftConnector interface
    # ------------------------------------------------------------------

    async def send_verification_outcome(
        self,
        batch_size: int,
        seq_ids: torch.Tensor,
        k_accepted: torch.Tensor,
        bonus_tokens: torch.Tensor,
        temperatures: torch.Tensor | None,
        hidden_states: torch.Tensor | None,
        aux_hidden_states: torch.Tensor | None,
        extend_counts: torch.Tensor | None,
        extend_hidden_states: torch.Tensor | None,
        extend_token_ids: torch.Tensor | None,
        needs_logits: bool = False,
    ) -> None:
        from vllm.v1.spec_decode.draft_data_models import (
            VerificationOutcome,
            encode_command,
        )

        if not self._connected:
            self._reconnect()
            if not self._connected:
                raise ConnectionError(f"Not connected to draft server at {self._address}")

        # Slice tensors to batch_size
        _seq_ids = seq_ids[:batch_size].to(torch.int64).reshape(-1)
        _k_accepted = k_accepted[:batch_size].to(torch.int64).reshape(-1)
        _bonus_tokens = bonus_tokens[:batch_size].to(torch.int64).reshape(-1)

        # Build ordered tensor list (must match draft server recv order)
        tensor_list: list[torch.Tensor] = [_seq_ids, _k_accepted, _bonus_tokens]

        temps_ref = None
        if temperatures is not None:
            _temps = temperatures[:batch_size].to(torch.float32)
            temps_ref = self._make_tensor_ref(_temps)
            tensor_list.append(_temps)

        hs_ref = None
        if hidden_states is not None:
            _hs = hidden_states[:batch_size]
            hs_ref = self._make_tensor_ref(_hs)
            tensor_list.append(_hs)

        aux_hs_ref = None
        if aux_hidden_states is not None:
            _aux = aux_hidden_states[:batch_size]
            aux_hs_ref = self._make_tensor_ref(_aux)
            tensor_list.append(_aux)

        ext_counts_ref = None
        if extend_counts is not None:
            _ec = extend_counts[:batch_size].to(torch.int64)
            ext_counts_ref = self._make_tensor_ref(_ec)
            tensor_list.append(_ec)

        ext_hs_ref = None
        if extend_hidden_states is not None:
            _ehs = extend_hidden_states[:batch_size]
            ext_hs_ref = self._make_tensor_ref(_ehs)
            tensor_list.append(_ehs)

        ext_ids_ref = None
        if extend_token_ids is not None:
            _eids = extend_token_ids[:batch_size].to(torch.int64)
            ext_ids_ref = self._make_tensor_ref(_eids)
            tensor_list.append(_eids)

        outcome = VerificationOutcome(
            verify_server_id=self._verify_server_id,
            batch_size=batch_size,
            seq_ids_ref=self._make_tensor_ref(_seq_ids),
            k_accepted_ref=self._make_tensor_ref(_k_accepted),
            bonus_tokens_ref=self._make_tensor_ref(_bonus_tokens),
            temperatures_ref=temps_ref,
            hidden_states_ref=hs_ref,
            aux_hidden_states_ref=aux_hs_ref,
            extend_counts_ref=ext_counts_ref,
            extend_hidden_states_ref=ext_hs_ref,
            extend_token_ids_ref=ext_ids_ref,
            needs_logits=needs_logits,
        )

        cmd_bytes = encode_command("SPECULATE", outcome)

        if self._pg is not None:
            # NCCL path: send metadata over ZMQ, tensors over NCCL
            await self._send_multipart(cmd_bytes, [])
            for t in tensor_list:
                self._nccl_send(t)
        else:
            # ZMQ-only path: single multipart message
            await self._send_multipart(cmd_bytes, tensor_list)

    async def recv_speculation(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        from vllm.v1.spec_decode.draft_data_models import (
            SpeculationResponse,
            decode,
        )

        if not self._connected:
            self._reconnect()
            if not self._connected:
                raise ConnectionError(f"Not connected to draft server at {self._address}")

        if self._pg is not None:
            # NCCL path: receive metadata over ZMQ, tensors over NCCL
            resp_bytes, _ = await self._recv_multipart()
            resp = decode(resp_bytes, SpeculationResponse)
            cache_hits = self._nccl_recv(
                resp.cache_hits_ref.shape,
                _str_to_dtype(resp.cache_hits_ref.dtype),
            )
            draft_tokens = self._nccl_recv(
                resp.draft_tokens_ref.shape,
                _str_to_dtype(resp.draft_tokens_ref.dtype),
            )
            draft_logits: torch.Tensor | None = None
            if resp.draft_logits_ref is not None:
                draft_logits = self._nccl_recv(
                    resp.draft_logits_ref.shape,
                    _str_to_dtype(resp.draft_logits_ref.dtype),
                )
        else:
            # ZMQ-only path: single multipart message
            resp_bytes, tensor_frames = await self._recv_multipart()
            resp = decode(resp_bytes, SpeculationResponse)

            idx = 0
            cache_hits = _bytes_to_tensor(
                tensor_frames[idx],
                resp.cache_hits_ref.shape,
                _str_to_dtype(resp.cache_hits_ref.dtype),
                self._device,
            )
            idx += 1
            draft_tokens = _bytes_to_tensor(
                tensor_frames[idx],
                resp.draft_tokens_ref.shape,
                _str_to_dtype(resp.draft_tokens_ref.dtype),
                self._device,
            )
            idx += 1
            draft_logits = None
            if resp.draft_logits_ref is not None:
                draft_logits = _bytes_to_tensor(
                    tensor_frames[idx],
                    resp.draft_logits_ref.shape,
                    _str_to_dtype(resp.draft_logits_ref.dtype),
                    self._device,
                )

        return cache_hits, draft_tokens, draft_logits

    async def send_and_recv_speculation(
        self,
        batch_size: int,
        seq_ids: torch.Tensor,
        k_accepted: torch.Tensor,
        bonus_tokens: torch.Tensor,
        temperatures: torch.Tensor | None,
        hidden_states: torch.Tensor | None,
        aux_hidden_states: torch.Tensor | None,
        extend_counts: torch.Tensor | None,
        extend_hidden_states: torch.Tensor | None,
        extend_token_ids: torch.Tensor | None,
        needs_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Send verification outcome and receive speculation in one async call.

        Combines send_verification_outcome + recv_speculation into a single
        coroutine to avoid two separate _run_async() event loop invocations.
        """
        await self.send_verification_outcome(
            batch_size=batch_size,
            seq_ids=seq_ids,
            k_accepted=k_accepted,
            bonus_tokens=bonus_tokens,
            temperatures=temperatures,
            hidden_states=hidden_states,
            aux_hidden_states=aux_hidden_states,
            extend_counts=extend_counts,
            extend_hidden_states=extend_hidden_states,
            extend_token_ids=extend_token_ids,
            needs_logits=needs_logits,
        )
        return await self.recv_speculation()

    async def send_prefill(
        self,
        seq_id: int,
        prompt_token_ids: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> None:
        from vllm.v1.spec_decode.draft_data_models import (
            PrefillRequest,
            encode_command,
        )

        if not self._connected:
            self._reconnect()
            if not self._connected:
                raise ConnectionError(f"Not connected to draft server at {self._address}")

        _prompt = prompt_token_ids.to(torch.int64)
        tensor_list: list[torch.Tensor] = [_prompt]

        hs_ref = None
        if hidden_states is not None:
            hs_ref = self._make_tensor_ref(hidden_states)
            tensor_list.append(hidden_states)

        prefill = PrefillRequest(
            verify_server_id=self._verify_server_id,
            seq_id=seq_id,
            prompt_token_ids_ref=self._make_tensor_ref(_prompt),
            hidden_states_ref=hs_ref,
        )

        cmd_bytes = encode_command("PREFILL", prefill)

        if self._pg is not None:
            await self._send_multipart(cmd_bytes, [])
            for t in tensor_list:
                self._nccl_send(t)
        else:
            await self._send_multipart(cmd_bytes, tensor_list)

    async def send_free_seq(self, seq_ids: torch.Tensor) -> None:
        from vllm.v1.spec_decode.draft_data_models import (
            FreeSeqRequest,
            encode_command,
        )

        if not self._connected:
            self._reconnect()
            if not self._connected:
                raise ConnectionError(f"Not connected to draft server at {self._address}")

        _seq_ids = seq_ids.to(torch.int64)
        seq_ids_ref = self._make_tensor_ref(_seq_ids)

        free_req = FreeSeqRequest(
            verify_server_id=self._verify_server_id,
            seq_ids_ref=seq_ids_ref,
        )

        cmd_bytes = encode_command("FREE_SEQ", free_req)

        if self._pg is not None:
            await self._send_multipart(cmd_bytes, [])
            self._nccl_send(_seq_ids)
        else:
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
        logger.info("ZmqNcclDraftConnector closed for %s", self._verify_server_id)


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
            logger.info("Draft server %s is reachable (TCP connect OK).", addr)
            reachable.append(addr)
        except Exception as exc:
            logger.warning("Draft server %s not reachable: %s", addr, exc)
            unreachable.append(addr)

    if not reachable:
        raise RuntimeError(
            f"No draft servers reachable at startup. "
            f"Tried {len(draft_addresses)} address(es): "
            f"{', '.join(draft_addresses)}. "
            f"Ensure draft servers are running and accessible."
        )
