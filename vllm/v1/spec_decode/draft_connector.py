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

import itertools
import logging
from abc import ABC, abstractmethod

import torch

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
        from vllm.v1.spec_decode.draft_data_models import (
            SpeculationResponse,
            VerificationOutcome,
            decode,
            encode_command,
        )

        self._ensure_connected()

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
        await self._send_multipart(cmd_bytes, tensor_list)

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
