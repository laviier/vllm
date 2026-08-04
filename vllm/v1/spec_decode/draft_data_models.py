# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data models for disaggregated speculative decoding (N:M).

Defines msgspec.Struct types for communication between verify servers
and draft servers. Metadata is serialized via msgspec/msgpack; tensor
payloads are transferred inline in the same ZMQ multipart message and
referenced by TensorRef.
"""

from typing import TypeVar

import msgspec

# Type variable for decode() generic return type
_T = TypeVar("_T")


class TensorRef(msgspec.Struct):
    """Reference to an inline tensor frame in the ZMQ multipart message."""

    shape: tuple[int, ...]
    dtype: str
    buffer_id: str
    nbytes: int


class VerificationOutcome(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    """Verify_Server → Draft_Server: results of last verification step."""

    verify_server_id: str
    batch_size: int
    seq_ids_ref: TensorRef  # [B] int64
    k_accepted_ref: TensorRef  # [B] int64
    bonus_tokens_ref: TensorRef  # [B] int64
    temperatures_ref: TensorRef | None = None  # [B] float32
    needs_logits: bool = False  # whether draft_logits should be returned


class SpeculationResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    """Draft_Server → Verify_Server: draft tokens for next decode step."""

    cache_hits_ref: TensorRef  # [B] bool
    draft_tokens_ref: TensorRef  # [B, K] int64
    draft_logits_ref: TensorRef | None = None  # [B, K, V]


class PrefillRequest(msgspec.Struct):
    """Verify_Server → Draft_Server: prefill a new request on draft model."""

    verify_server_id: str
    seq_id: int
    prompt_token_ids_ref: TensorRef


class FreeSeqRequest(msgspec.Struct):
    """Verify_Server → Draft_Server: free KV cache for completed requests."""

    verify_server_id: str
    seq_ids_ref: TensorRef  # [N] int64


class IpcHandshake(msgspec.Struct):
    """Verify_Server → Draft_Server: attach a CUDA-IPC SPECULATE ring
    buffer alongside the existing ZMQ channel.

    Sent once per connection at startup. Draft server opens the GPU
    handles + doorbell shm and services SPECULATEs via the ring;
    PREFILL/FREE_SEQ still travel via ZMQ.
    """

    verify_server_id: str
    shm_path: str
    max_batch: int
    K: int
    n_slots: int
    # Pickle of {tensor_name: (rebuild_fn, args)} — torch.mp.reductions
    # output. Kept as opaque bytes so msgspec doesn't need to know the
    # torch reduction schema.
    gpu_handles_pickle: bytes


class IpcHandshakeAck(msgspec.Struct):
    """Draft_Server → Verify_Server: acknowledgement of IPC_HANDSHAKE."""

    ok: bool
    error: str = ""


class DraftCommand(msgspec.Struct):
    """Envelope for all draft service messages over ZMQ."""

    command: str  # "SPECULATE", "PREFILL", "FREE_SEQ", "EXIT",
    # "HEALTHCHECK", "IPC_HANDSHAKE"
    payload: bytes  # msgspec-encoded inner message


# Single shared encoder (thread-safe for encoding)
_encoder = msgspec.msgpack.Encoder()

# Per-type decoders — msgspec Decoder is fastest when bound to a concrete type
_decoders: dict[type, msgspec.msgpack.Decoder] = {
    TensorRef: msgspec.msgpack.Decoder(TensorRef),
    VerificationOutcome: msgspec.msgpack.Decoder(VerificationOutcome),
    SpeculationResponse: msgspec.msgpack.Decoder(SpeculationResponse),
    PrefillRequest: msgspec.msgpack.Decoder(PrefillRequest),
    FreeSeqRequest: msgspec.msgpack.Decoder(FreeSeqRequest),
    IpcHandshake: msgspec.msgpack.Decoder(IpcHandshake),
    IpcHandshakeAck: msgspec.msgpack.Decoder(IpcHandshakeAck),
    DraftCommand: msgspec.msgpack.Decoder(DraftCommand),
}

DraftMessage = (
    TensorRef
    | VerificationOutcome
    | SpeculationResponse
    | PrefillRequest
    | FreeSeqRequest
    | IpcHandshake
    | IpcHandshakeAck
    | DraftCommand
)


def encode(msg: DraftMessage) -> bytes:
    """Serialize a draft data model to msgpack bytes."""
    return _encoder.encode(msg)


def decode(data: bytes, msg_type: type[_T]) -> _T:
    """Deserialize msgpack bytes into the specified draft data model type."""
    return _decoders[msg_type].decode(data)


def encode_command(
    command: str,
    payload_msg: (
        VerificationOutcome
        | SpeculationResponse
        | PrefillRequest
        | FreeSeqRequest
        | IpcHandshake
        | IpcHandshakeAck
        | None
    ) = None,
) -> bytes:
    """Build and serialize a :class:`DraftCommand` envelope."""
    payload = _encoder.encode(payload_msg) if payload_msg is not None else b""
    return _encoder.encode(DraftCommand(command=command, payload=payload))


def decode_command(data: bytes) -> DraftCommand:
    """Deserialize a :class:`DraftCommand` envelope from msgpack bytes."""
    return _decoders[DraftCommand].decode(data)
