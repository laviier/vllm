# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire transport for ``DraftServer`` over ZMQ multipart frames.

Decoding inbound tensor frames into device tensors and encoding
outbound responses (with TensorRef metadata + raw tensor bytes).
Also handles the SPECULATE-specific tensor sequence read.

Expects the consumer to initialise:

    self._socket  (ZMQ ROUTER)
    self._current_tensor_frames: list[bytes]
    self._current_tensor_idx: int
    self._buffer_counter: itertools.count
    self.device, self.dtype, self.K
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.v1.spec_decode.draft_connector import (
    _dtype_to_str,
    _str_to_dtype,
    _tensor_to_bytes,
)
from vllm.v1.spec_decode.draft_data_models import (
    SpeculationResponse,
    TensorRef,
    VerificationOutcome,
    encode,
)

logger = init_logger(__name__)


class DraftServerTransportMixin:
    """Mixin: wire transport for ``DraftServer``."""

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
