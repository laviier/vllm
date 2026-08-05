# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioral tests for the disaggregated draft connector."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from vllm.v1.spec_decode.draft_connector import (
    CudaIpcDraftConnector,
    EagleTargetInputs,
    ZmqDraftConnector,
    _dtype_to_str,
    _IpcSharedBuffer,
    _str_to_dtype,
    _tensor_to_bytes,
)
from vllm.v1.spec_decode.draft_data_models import (
    FreeSeqRequest,
    PrefillRequest,
    SpeculationResponse,
    VerificationOutcome,
    decode,
    decode_command,
    encode,
)


@pytest.fixture
def connector() -> ZmqDraftConnector:
    connector = ZmqDraftConnector.__new__(ZmqDraftConnector)
    connector._address = "tcp://127.0.0.1:50051"
    connector._verify_server_id = "vs-test"
    connector._device = torch.device("cpu")
    connector._timeout_ms = 5000
    connector._buffer_counter = iter(range(100))
    connector._connected = True
    connector._socket = MagicMock()
    connector._zmq_ctx = MagicMock()
    return connector


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().numpy().tobytes()


def test_dtype_helpers():
    assert _dtype_to_str(torch.float32) == "float32"
    assert _str_to_dtype("int64") == torch.int64


def test_bfloat16_wire_bytes_preserve_dtype_and_size():
    tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    data = _tensor_to_bytes(tensor)
    assert len(data) == tensor.numel() * tensor.element_size()
    decoded = torch.frombuffer(bytearray(data), dtype=torch.bfloat16)
    assert torch.equal(decoded, tensor)


def test_tensor_refs_are_unique(connector: ZmqDraftConnector):
    tensor = torch.zeros((2, 3), dtype=torch.float32)
    first = connector._make_tensor_ref(tensor)
    second = connector._make_tensor_ref(tensor)

    assert first.shape == (2, 3)
    assert first.dtype == "float32"
    assert first.nbytes == 24
    assert first.buffer_id != second.buffer_id


def test_reconnects_when_needed(connector: ZmqDraftConnector):
    connector._connected = False
    with patch.object(connector, "_reconnect") as reconnect:
        reconnect.side_effect = lambda: setattr(connector, "_connected", True)
        connector._ensure_connected()
    reconnect.assert_called_once()


def test_raises_when_reconnect_fails(connector: ZmqDraftConnector):
    connector._connected = False
    with (
        patch.object(connector, "_reconnect"),
        pytest.raises(ConnectionError, match="Not connected"),
    ):
        connector._ensure_connected()


def test_close_releases_resources(connector: ZmqDraftConnector):
    socket = connector._socket
    context = connector._zmq_ctx
    connector.close()

    assert not connector.connected
    socket.close.assert_called_once_with(linger=0)
    context.term.assert_called_once()
    assert connector._socket is None
    assert connector._zmq_ctx is None


@pytest.mark.anyio
async def test_send_and_recv_speculation(connector: ZmqDraftConnector):
    cache_hits = torch.tensor([True, False])
    draft_tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    response = SpeculationResponse(
        cache_hits_ref=connector._make_tensor_ref(cache_hits),
        draft_tokens_ref=connector._make_tensor_ref(draft_tokens),
    )
    connector._send_multipart = AsyncMock()
    connector._recv_multipart = AsyncMock(
        return_value=(
            encode(response),
            [_tensor_bytes(cache_hits), _tensor_bytes(draft_tokens)],
        )
    )

    result = await connector.send_and_recv_speculation(
        batch_size=2,
        seq_ids=torch.tensor([10, 11]),
        k_accepted=torch.tensor([2, 1]),
        bonus_tokens=torch.tensor([20, 21]),
        temperatures=torch.tensor([0.5, 0.7]),
        needs_logits=True,
    )

    sent_metadata, sent_tensors = connector._send_multipart.await_args.args
    command = decode_command(sent_metadata)
    outcome = decode(command.payload, VerificationOutcome)
    assert command.command == "SPECULATE"
    assert outcome.verify_server_id == "vs-test"
    assert outcome.batch_size == 2
    assert outcome.needs_logits
    assert outcome.temperatures_ref is not None
    assert len(sent_tensors) == 4
    assert torch.equal(result[0], cache_hits)
    assert torch.equal(result[1], draft_tokens)
    assert result[2] is None


@pytest.mark.anyio
async def test_send_and_recv_eagle_speculation(connector: ZmqDraftConnector):
    cache_hits = torch.tensor([True])
    draft_tokens = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    response = SpeculationResponse(
        cache_hits_ref=connector._make_tensor_ref(cache_hits),
        draft_tokens_ref=connector._make_tensor_ref(draft_tokens),
    )
    connector._send_multipart = AsyncMock()
    connector._recv_multipart = AsyncMock(
        return_value=(
            encode(response),
            [_tensor_bytes(cache_hits), _tensor_bytes(draft_tokens)],
        )
    )
    eagle_inputs = EagleTargetInputs(
        token_ids=torch.tensor([10, 11], dtype=torch.int32),
        positions=torch.tensor([0, 1], dtype=torch.int64),
        query_lens=torch.tensor([2], dtype=torch.int32),
        hidden_states=torch.ones((2, 4), dtype=torch.bfloat16),
    )

    await connector.send_and_recv_speculation(
        batch_size=1,
        seq_ids=torch.tensor([7]),
        k_accepted=torch.tensor([0]),
        bonus_tokens=torch.tensor([12]),
        temperatures=None,
        eagle_inputs=eagle_inputs,
    )

    sent_metadata, sent_tensors = connector._send_multipart.await_args.args
    outcome = decode(
        decode_command(sent_metadata).payload,
        VerificationOutcome,
    )
    assert outcome.eagle_token_ids_ref is not None
    assert outcome.eagle_positions_ref is not None
    assert outcome.eagle_query_lens_ref is not None
    assert outcome.eagle_hidden_states_ref is not None
    assert [tensor.shape for tensor in sent_tensors[3:]] == [
        torch.Size([2]),
        torch.Size([2]),
        torch.Size([1]),
        torch.Size([2, 4]),
    ]


@pytest.mark.anyio
async def test_send_and_recv_speculation_with_logits(
    connector: ZmqDraftConnector,
):
    cache_hits = torch.tensor([True])
    draft_tokens = torch.tensor([[1, 2]], dtype=torch.int64)
    draft_logits = torch.ones((1, 2, 4), dtype=torch.float32)
    response = SpeculationResponse(
        cache_hits_ref=connector._make_tensor_ref(cache_hits),
        draft_tokens_ref=connector._make_tensor_ref(draft_tokens),
        draft_logits_ref=connector._make_tensor_ref(draft_logits),
    )
    connector._send_multipart = AsyncMock()
    connector._recv_multipart = AsyncMock(
        return_value=(
            encode(response),
            [
                _tensor_bytes(cache_hits),
                _tensor_bytes(draft_tokens),
                _tensor_bytes(draft_logits),
            ],
        )
    )

    _, _, result_logits = await connector.send_and_recv_speculation(
        batch_size=1,
        seq_ids=torch.tensor([10]),
        k_accepted=torch.tensor([2]),
        bonus_tokens=torch.tensor([20]),
        temperatures=None,
    )
    assert result_logits is not None
    assert torch.equal(result_logits, draft_logits)


@pytest.mark.anyio
async def test_send_prefill(connector: ZmqDraftConnector):
    connector._send_multipart = AsyncMock()
    prompt = torch.tensor([1, 2, 3])
    await connector.send_prefill(42, prompt)

    metadata, tensors = connector._send_multipart.await_args.args
    command = decode_command(metadata)
    request = decode(command.payload, PrefillRequest)
    assert command.command == "PREFILL"
    assert request.verify_server_id == "vs-test"
    assert request.seq_id == 42
    assert torch.equal(tensors[0], prompt)


@pytest.mark.anyio
async def test_send_free_seq(connector: ZmqDraftConnector):
    connector._send_multipart = AsyncMock()
    seq_ids = torch.tensor([3, 4])
    await connector.send_free_seq(seq_ids)

    metadata, tensors = connector._send_multipart.await_args.args
    command = decode_command(metadata)
    request = decode(command.payload, FreeSeqRequest)
    assert command.command == "FREE_SEQ"
    assert request.verify_server_id == "vs-test"
    assert torch.equal(tensors[0], seq_ids)


@pytest.mark.anyio
async def test_default_dispatch_defers_roundtrip(connector: ZmqDraftConnector):
    expected = (torch.ones(1, dtype=torch.bool), torch.ones((1, 2)), None)
    connector.send_and_recv_speculation = AsyncMock(return_value=expected)
    handle = connector.dispatch_speculation(
        batch_size=1,
        seq_ids=torch.tensor([1]),
        k_accepted=torch.tensor([0]),
        bonus_tokens=torch.tensor([2]),
        temperatures=None,
    )

    assert not connector.send_and_recv_speculation.called
    assert await connector.await_speculation(handle) == expected
    connector.send_and_recv_speculation.assert_awaited_once()


def test_cuda_ipc_dispatch_copies_eagle_payload():
    connector = CudaIpcDraftConnector.__new__(CudaIpcDraftConnector)
    connector._ipc_ready = True
    connector._ipc_max_batch = 4
    connector._ipc_K = 3
    connector._eagle_max_tokens = 8
    connector._eagle_hidden_size = 4
    connector._next_slot = 0
    connector._slot_seqs = [0] * 16
    connector._buf = _IpcSharedBuffer(
        torch.device("cpu"),
        max_batch=4,
        K=3,
        eagle_max_tokens=8,
        eagle_hidden_size=4,
    )
    eagle_inputs = EagleTargetInputs(
        token_ids=torch.tensor([10, 11], dtype=torch.int32),
        positions=torch.tensor([2, 3], dtype=torch.int64),
        query_lens=torch.tensor([2], dtype=torch.int32),
        hidden_states=torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
    )

    handle = connector.dispatch_speculation(
        batch_size=1,
        seq_ids=torch.tensor([7]),
        k_accepted=torch.tensor([0]),
        bonus_tokens=torch.tensor([12]),
        temperatures=None,
        eagle_inputs=eagle_inputs,
    )

    assert handle.state == {
        "slot": 0,
        "seq16": 1,
        "batch_size": 1,
        "is_eagle": True,
    }
    assert connector._buf.req_eagle_num_tokens[0].item() == 2
    assert torch.equal(
        connector._buf.req_eagle_token_ids[:2],
        eagle_inputs.token_ids,
    )
    assert torch.equal(
        connector._buf.req_eagle_positions[:2],
        eagle_inputs.positions,
    )
    assert torch.equal(
        connector._buf.req_eagle_query_lens[:1],
        eagle_inputs.query_lens,
    )
    assert torch.equal(
        connector._buf.req_eagle_hidden_states[:2],
        eagle_inputs.hidden_states,
    )
