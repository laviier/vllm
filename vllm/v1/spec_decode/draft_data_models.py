# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Data models for disaggregated speculative decoding (N:M).

Defines msgspec.Struct types for communication between Verify_Servers
and Draft_Servers. Metadata is serialized via msgspec/msgpack; tensor
payloads are transferred out-of-band (NCCL/TCP) and referenced by
TensorRef.
"""

from typing import TypeVar, Union

import msgspec

# Type variable for decode() generic return type
_T = TypeVar("_T")


class TensorRef(msgspec.Struct):
    """Reference to an out-of-band tensor (NCCL/TCP/shared memory)."""

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
    hidden_states_ref: TensorRef | None = None  # [B, hidden_size]
    aux_hidden_states_ref: TensorRef | None = None  # [B, aux_hidden_size]
    extend_counts_ref: TensorRef | None = None  # [B] int64
    extend_hidden_states_ref: TensorRef | None = None  # [B, K, hidden_size]
    extend_token_ids_ref: TensorRef | None = None  # [B, K] int64
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
    hidden_states_ref: TensorRef | None = None


class FreeSeqRequest(msgspec.Struct):
    """Verify_Server → Draft_Server: free KV cache for completed requests."""

    verify_server_id: str
    seq_ids_ref: TensorRef  # [N] int64


class DraftCommand(msgspec.Struct):
    """Envelope for all draft service messages over ZMQ."""

    command: str  # "SPECULATE", "PREFILL", "FREE_SEQ", "EXIT", "HEALTHCHECK"
    payload: bytes  # msgspec-encoded inner message


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
# Module-level encoder/decoder instances following the same pattern as
# existing P/D connectors (Mooncake, NIXL, MoRIIO).  The Encoder is
# type-agnostic; each message type gets its own typed Decoder for
# zero-copy deserialization.
# ---------------------------------------------------------------------------

# Single shared encoder (thread-safe for encoding)
_encoder = msgspec.msgpack.Encoder()

# Per-type decoders — msgspec Decoder is fastest when bound to a concrete type
_decoders: dict[type, msgspec.msgpack.Decoder] = {
    TensorRef: msgspec.msgpack.Decoder(TensorRef),
    VerificationOutcome: msgspec.msgpack.Decoder(VerificationOutcome),
    SpeculationResponse: msgspec.msgpack.Decoder(SpeculationResponse),
    PrefillRequest: msgspec.msgpack.Decoder(PrefillRequest),
    FreeSeqRequest: msgspec.msgpack.Decoder(FreeSeqRequest),
    DraftCommand: msgspec.msgpack.Decoder(DraftCommand),
}

# Union of all message types that can be encoded/decoded
DraftMessage = Union[
    TensorRef,
    VerificationOutcome,
    SpeculationResponse,
    PrefillRequest,
    FreeSeqRequest,
    DraftCommand,
]


def encode(msg: DraftMessage) -> bytes:
    """Serialize a draft data model to msgpack bytes.

    Tensor data is NOT included — only ``TensorRef`` metadata is
    serialized.  Actual tensor payloads are transferred out-of-band
    via NCCL or TCP.

    Args:
        msg: Any of the draft data model ``msgspec.Struct`` instances.

    Returns:
        Msgpack-encoded bytes.
    """
    return _encoder.encode(msg)


def decode(data: bytes, msg_type: type[_T]) -> _T:
    """Deserialize msgpack bytes into the specified draft data model type.

    Uses pre-built typed decoders for each message type, matching the
    pattern used by the Mooncake and NIXL connectors.

    Args:
        data: Msgpack-encoded bytes produced by :func:`encode`.
        msg_type: The target ``msgspec.Struct`` class to decode into.

    Returns:
        An instance of *msg_type*.

    Raises:
        KeyError: If *msg_type* is not a known draft data model.
        msgspec.DecodeError: If *data* cannot be decoded into *msg_type*.
    """
    return _decoders[msg_type].decode(data)


def encode_command(
    command: str,
    payload_msg: Union[
        VerificationOutcome,
        SpeculationResponse,
        PrefillRequest,
        FreeSeqRequest,
        None,
    ] = None,
) -> bytes:
    """Build and serialize a :class:`DraftCommand` envelope.

    This implements the command-envelope pattern used by the Draft_Server
    protocol: the outer ``DraftCommand`` carries a command string and an
    inner msgspec-encoded payload.

    Args:
        command: One of ``"SPECULATE"``, ``"PREFILL"``, ``"FREE_SEQ"``,
            ``"EXIT"``, ``"HEALTHCHECK"``.
        payload_msg: The inner message to encode into the envelope's
            ``payload`` field.  May be ``None`` for commands that carry
            no payload (e.g. ``"EXIT"``, ``"HEALTHCHECK"``).

    Returns:
        Msgpack-encoded bytes of the ``DraftCommand`` envelope.
    """
    payload = _encoder.encode(payload_msg) if payload_msg is not None else b""
    return _encoder.encode(DraftCommand(command=command, payload=payload))


def decode_command(data: bytes) -> DraftCommand:
    """Deserialize a :class:`DraftCommand` envelope from msgpack bytes.

    After decoding the envelope, callers should use :func:`decode` on
    ``cmd.payload`` with the appropriate inner message type based on
    ``cmd.command``.

    Args:
        data: Msgpack-encoded bytes of a ``DraftCommand``.

    Returns:
        The decoded ``DraftCommand`` instance.
    """
    return _decoders[DraftCommand].decode(data)
