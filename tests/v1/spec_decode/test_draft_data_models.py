# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for draft data model serialization helpers."""

import pytest

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
    encode_command,
)


def _make_tensor_ref(**overrides):
    defaults = dict(shape=(4,), dtype="int64", buffer_id="buf-0", nbytes=32)
    defaults.update(overrides)
    return TensorRef(**defaults)


class TestEncodeDecodeRoundTrip:
    """Verify encode → decode round-trip for every data model type."""

    def test_tensor_ref(self):
        ref = _make_tensor_ref(shape=(2, 3), dtype="float32", nbytes=24)
        assert decode(encode(ref), TensorRef) == ref

    def test_verification_outcome_required_only(self):
        msg = VerificationOutcome(
            verify_server_id="vs-1",
            batch_size=4,
            seq_ids_ref=_make_tensor_ref(),
            k_accepted_ref=_make_tensor_ref(buffer_id="buf-1"),
            bonus_tokens_ref=_make_tensor_ref(buffer_id="buf-2"),
        )
        assert decode(encode(msg), VerificationOutcome) == msg

    def test_verification_outcome_all_optional_fields(self):
        msg = VerificationOutcome(
            verify_server_id="vs-2",
            batch_size=8,
            seq_ids_ref=_make_tensor_ref(),
            k_accepted_ref=_make_tensor_ref(buffer_id="buf-1"),
            bonus_tokens_ref=_make_tensor_ref(buffer_id="buf-2"),
            temperatures_ref=_make_tensor_ref(
                dtype="float32", buffer_id="buf-3"
            ),
            hidden_states_ref=_make_tensor_ref(
                shape=(8, 4096), dtype="float16", buffer_id="buf-4",
                nbytes=65536,
            ),
            aux_hidden_states_ref=_make_tensor_ref(
                shape=(8, 1024), buffer_id="buf-5"
            ),
            extend_counts_ref=_make_tensor_ref(buffer_id="buf-6"),
            extend_hidden_states_ref=_make_tensor_ref(
                shape=(8, 5, 4096), buffer_id="buf-7"
            ),
            extend_token_ids_ref=_make_tensor_ref(
                shape=(8, 5), buffer_id="buf-8"
            ),
        )
        assert decode(encode(msg), VerificationOutcome) == msg

    def test_speculation_response_required_only(self):
        msg = SpeculationResponse(
            cache_hits_ref=_make_tensor_ref(dtype="bool", buffer_id="ch"),
            draft_tokens_ref=_make_tensor_ref(
                shape=(4, 5), buffer_id="dt"
            ),
        )
        assert decode(encode(msg), SpeculationResponse) == msg

    def test_speculation_response_with_logits(self):
        msg = SpeculationResponse(
            cache_hits_ref=_make_tensor_ref(dtype="bool", buffer_id="ch"),
            draft_tokens_ref=_make_tensor_ref(
                shape=(4, 5), buffer_id="dt"
            ),
            draft_logits_ref=_make_tensor_ref(
                shape=(4, 5, 32000), dtype="float16", buffer_id="dl",
                nbytes=2048000,
            ),
        )
        assert decode(encode(msg), SpeculationResponse) == msg

    def test_prefill_request(self):
        msg = PrefillRequest(
            verify_server_id="vs-1",
            seq_id=42,
            prompt_token_ids_ref=_make_tensor_ref(
                shape=(128,), buffer_id="prompt"
            ),
            hidden_states_ref=_make_tensor_ref(
                shape=(128, 4096), buffer_id="hs"
            ),
        )
        assert decode(encode(msg), PrefillRequest) == msg

    def test_prefill_request_no_hidden_states(self):
        msg = PrefillRequest(
            verify_server_id="vs-1",
            seq_id=7,
            prompt_token_ids_ref=_make_tensor_ref(shape=(64,)),
        )
        assert decode(encode(msg), PrefillRequest) == msg

    def test_free_seq_request(self):
        msg = FreeSeqRequest(
            verify_server_id="vs-3",
            seq_ids_ref=_make_tensor_ref(shape=(3,), buffer_id="free"),
        )
        assert decode(encode(msg), FreeSeqRequest) == msg

    def test_draft_command(self):
        inner = encode(FreeSeqRequest(
            verify_server_id="vs-1",
            seq_ids_ref=_make_tensor_ref(),
        ))
        cmd = DraftCommand(command="FREE_SEQ", payload=inner)
        assert decode(encode(cmd), DraftCommand) == cmd


class TestCommandEnvelope:
    """Verify the DraftCommand envelope encode/decode helpers."""

    def test_encode_decode_command_with_payload(self):
        outcome = VerificationOutcome(
            verify_server_id="vs-1",
            batch_size=2,
            seq_ids_ref=_make_tensor_ref(),
            k_accepted_ref=_make_tensor_ref(buffer_id="k"),
            bonus_tokens_ref=_make_tensor_ref(buffer_id="b"),
        )
        data = encode_command("SPECULATE", outcome)
        cmd = decode_command(data)
        assert cmd.command == "SPECULATE"
        # Inner payload should decode back to the original message
        inner = decode(cmd.payload, VerificationOutcome)
        assert inner == outcome

    def test_encode_decode_command_no_payload(self):
        data = encode_command("EXIT")
        cmd = decode_command(data)
        assert cmd.command == "EXIT"
        assert cmd.payload == b""

    def test_encode_decode_command_healthcheck(self):
        data = encode_command("HEALTHCHECK")
        cmd = decode_command(data)
        assert cmd.command == "HEALTHCHECK"
        assert cmd.payload == b""

    def test_encode_decode_prefill_command(self):
        prefill = PrefillRequest(
            verify_server_id="vs-2",
            seq_id=99,
            prompt_token_ids_ref=_make_tensor_ref(shape=(256,)),
        )
        data = encode_command("PREFILL", prefill)
        cmd = decode_command(data)
        assert cmd.command == "PREFILL"
        inner = decode(cmd.payload, PrefillRequest)
        assert inner == prefill

    def test_encode_decode_free_seq_command(self):
        free = FreeSeqRequest(
            verify_server_id="vs-1",
            seq_ids_ref=_make_tensor_ref(shape=(5,)),
        )
        data = encode_command("FREE_SEQ", free)
        cmd = decode_command(data)
        assert cmd.command == "FREE_SEQ"
        inner = decode(cmd.payload, FreeSeqRequest)
        assert inner == free


class TestDecodeErrors:
    """Verify error handling for invalid inputs."""

    def test_unknown_type_raises_key_error(self):
        with pytest.raises(KeyError):
            decode(b"\x00", int)  # type: ignore[arg-type]

    def test_corrupted_data_raises_decode_error(self):
        import msgspec
        with pytest.raises(msgspec.DecodeError):
            decode(b"\xff\xff\xff", TensorRef)
