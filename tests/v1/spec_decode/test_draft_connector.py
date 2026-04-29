# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ZmqNcclDraftConnector.

Tests the connector's ZMQ metadata flow, TensorRef construction,
connection management, and error handling. NCCL tensor transport
is mocked since it requires actual GPU process groups.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.v1.spec_decode.draft_connector import (
    ZmqNcclDraftConnector,
    _dtype_to_str,
    _str_to_dtype,
)
from vllm.v1.spec_decode.draft_data_models import (
    SpeculationResponse,
    TensorRef,
    VerificationOutcome,
    decode,
    decode_command,
    encode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_pg():
    """Create a mock NCCL process group."""
    pg = MagicMock()
    work = MagicMock()
    work.wait = MagicMock()
    pg.send = MagicMock(return_value=work)
    pg.recv = MagicMock(return_value=work)
    return pg


def _make_connector(
    address: str = "tcp://127.0.0.1:50051",
    verify_server_id: str = "vs-test-1",
    timeout_ms: int = 5000,
) -> ZmqNcclDraftConnector:
    """Create a ZmqNcclDraftConnector with mocked ZMQ and NCCL."""
    pg = _make_mock_pg()
    device = torch.device("cpu")

    # Patch make_zmq_socket where it's imported (inside _connect)
    with patch("vllm.utils.network_utils.make_zmq_socket") as mock_make:
        mock_socket = MagicMock()
        mock_socket.setsockopt = MagicMock()
        mock_socket.close = MagicMock()
        mock_make.return_value = mock_socket

        connector = ZmqNcclDraftConnector(
            address=address,
            verify_server_id=verify_server_id,
            process_group=pg,
            peer_rank=1,
            device=device,
            timeout_ms=timeout_ms,
        )
    return connector


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------


class TestDtypeHelpers:
    def test_dtype_to_str(self):
        assert _dtype_to_str(torch.float32) == "float32"
        assert _dtype_to_str(torch.int64) == "int64"
        assert _dtype_to_str(torch.bfloat16) == "bfloat16"

    def test_str_to_dtype(self):
        assert _str_to_dtype("float32") == torch.float32
        assert _str_to_dtype("int64") == torch.int64
        assert _str_to_dtype("bool") == torch.bool


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


class TestConnectionManagement:
    def test_initial_connection(self):
        connector = _make_connector()
        assert connector.connected is True

    def test_close_sets_disconnected(self):
        connector = _make_connector()
        connector.close()
        assert connector.connected is False
        assert connector._socket is None
        assert connector._zmq_ctx is None

    def test_reconnect_on_failure(self):
        connector = _make_connector()
        connector._connected = False

        with patch.object(connector, "_connect") as mock_connect:
            connector._reconnect()
            mock_connect.assert_called_once()

    def test_reconnect_failure_stays_disconnected(self):
        connector = _make_connector()
        connector._connected = False

        with patch.object(
            connector, "_connect", side_effect=Exception("fail")
        ):
            connector._reconnect()
            assert connector.connected is False


# ---------------------------------------------------------------------------
# TensorRef construction
# ---------------------------------------------------------------------------


class TestTensorRefConstruction:
    def test_make_tensor_ref_shape_and_dtype(self):
        connector = _make_connector()
        t = torch.zeros(4, 8, dtype=torch.float32)
        ref = connector._make_tensor_ref(t)

        assert ref.shape == (4, 8)
        assert ref.dtype == "float32"
        assert ref.nbytes == 4 * 8 * 4  # 4*8 elements * 4 bytes

    def test_buffer_ids_are_unique(self):
        connector = _make_connector()
        t = torch.zeros(2, dtype=torch.int64)
        ref1 = connector._make_tensor_ref(t)
        ref2 = connector._make_tensor_ref(t)
        assert ref1.buffer_id != ref2.buffer_id

    def test_make_tensor_ref_1d(self):
        connector = _make_connector()
        t = torch.zeros(10, dtype=torch.int64)
        ref = connector._make_tensor_ref(t)
        assert ref.shape == (10,)
        assert ref.dtype == "int64"
        assert ref.nbytes == 10 * 8


# ---------------------------------------------------------------------------
# send_verification_outcome metadata
# ---------------------------------------------------------------------------


class TestSendVerificationOutcome:
    @pytest.mark.anyio
    async def test_sends_command_envelope(self):
        connector = _make_connector()

        # Capture what gets sent over ZMQ
        sent_data = []

        async def mock_send(data):
            sent_data.append(data)

        connector._zmq_send = mock_send  # type: ignore[assignment]
        connector._nccl_send = MagicMock()  # mock NCCL

        B = 2
        await connector.send_verification_outcome(
            batch_size=B,
            seq_ids=torch.tensor([1, 2], dtype=torch.int64),
            k_accepted=torch.tensor([3, 4], dtype=torch.int64),
            bonus_tokens=torch.tensor([5, 6], dtype=torch.int64),
            temperatures=None,
            hidden_states=None,
            aux_hidden_states=None,
            extend_counts=None,
            extend_hidden_states=None,
            extend_token_ids=None,
        )

        # Should have sent exactly one ZMQ message
        assert len(sent_data) == 1

        # Decode the command envelope
        cmd = decode_command(sent_data[0])
        assert cmd.command == "SPECULATE"

        # Decode the inner payload
        outcome = decode(cmd.payload, VerificationOutcome)
        assert outcome.verify_server_id == "vs-test-1"
        assert outcome.batch_size == B
        assert outcome.temperatures_ref is None
        assert outcome.hidden_states_ref is None

    @pytest.mark.anyio
    async def test_sends_nccl_tensors_in_order(self):
        connector = _make_connector()

        async def mock_zmq_send(data):
            pass

        connector._zmq_send = mock_zmq_send  # type: ignore[assignment]

        nccl_calls = []

        def tracking_nccl_send(tensor):
            nccl_calls.append(tensor.shape)

        connector._nccl_send = tracking_nccl_send  # type: ignore[assignment]

        B = 3
        await connector.send_verification_outcome(
            batch_size=B,
            seq_ids=torch.zeros(B, dtype=torch.int64),
            k_accepted=torch.zeros(B, dtype=torch.int64),
            bonus_tokens=torch.zeros(B, dtype=torch.int64),
            temperatures=torch.zeros(B, dtype=torch.float32),
            hidden_states=torch.zeros(B, 128, dtype=torch.float16),
            aux_hidden_states=None,
            extend_counts=None,
            extend_hidden_states=None,
            extend_token_ids=None,
        )

        # Required: seq_ids, k_accepted, bonus_tokens
        # Optional present: temperatures, hidden_states
        assert len(nccl_calls) == 5
        assert nccl_calls[0] == (B,)  # seq_ids
        assert nccl_calls[1] == (B,)  # k_accepted
        assert nccl_calls[2] == (B,)  # bonus_tokens
        assert nccl_calls[3] == (B,)  # temperatures
        assert nccl_calls[4] == (B, 128)  # hidden_states

    @pytest.mark.anyio
    async def test_raises_when_disconnected(self):
        connector = _make_connector()
        connector._connected = False

        # Mock _reconnect to keep it disconnected
        connector._reconnect = MagicMock()  # type: ignore[assignment]
        connector._connected = False

        with pytest.raises(ConnectionError):
            await connector.send_verification_outcome(
                batch_size=1,
                seq_ids=torch.tensor([1], dtype=torch.int64),
                k_accepted=torch.tensor([0], dtype=torch.int64),
                bonus_tokens=torch.tensor([0], dtype=torch.int64),
                temperatures=None,
                hidden_states=None,
                aux_hidden_states=None,
                extend_counts=None,
                extend_hidden_states=None,
                extend_token_ids=None,
            )


# ---------------------------------------------------------------------------
# send_prefill metadata
# ---------------------------------------------------------------------------


class TestSendPrefill:
    @pytest.mark.anyio
    async def test_sends_prefill_command(self):
        connector = _make_connector()

        sent_data = []

        async def mock_send(data):
            sent_data.append(data)

        connector._zmq_send = mock_send  # type: ignore[assignment]
        connector._nccl_send = MagicMock()

        await connector.send_prefill(
            seq_id=42,
            prompt_token_ids=torch.tensor([1, 2, 3], dtype=torch.int64),
            hidden_states=None,
        )

        cmd = decode_command(sent_data[0])
        assert cmd.command == "PREFILL"

    @pytest.mark.anyio
    async def test_sends_hidden_states_when_present(self):
        connector = _make_connector()

        async def mock_zmq_send(data):
            pass

        connector._zmq_send = mock_zmq_send  # type: ignore[assignment]

        nccl_calls = []

        def tracking_nccl_send(tensor):
            nccl_calls.append(tensor.shape)

        connector._nccl_send = tracking_nccl_send  # type: ignore[assignment]

        await connector.send_prefill(
            seq_id=1,
            prompt_token_ids=torch.zeros(10, dtype=torch.int64),
            hidden_states=torch.zeros(10, 256, dtype=torch.float16),
        )

        # prompt_token_ids + hidden_states
        assert len(nccl_calls) == 2
        assert nccl_calls[0] == (10,)
        assert nccl_calls[1] == (10, 256)


# ---------------------------------------------------------------------------
# send_free_seq metadata
# ---------------------------------------------------------------------------


class TestSendFreeSeq:
    @pytest.mark.anyio
    async def test_sends_free_seq_command(self):
        connector = _make_connector()

        sent_data = []

        async def mock_send(data):
            sent_data.append(data)

        connector._zmq_send = mock_send  # type: ignore[assignment]
        connector._nccl_send = MagicMock()

        await connector.send_free_seq(
            seq_ids=torch.tensor([1, 2, 3], dtype=torch.int64)
        )

        cmd = decode_command(sent_data[0])
        assert cmd.command == "FREE_SEQ"

    @pytest.mark.anyio
    async def test_sends_seq_ids_via_nccl(self):
        connector = _make_connector()

        async def mock_zmq_send(data):
            pass

        connector._zmq_send = mock_zmq_send  # type: ignore[assignment]

        nccl_calls = []

        def tracking_nccl_send(tensor):
            nccl_calls.append(tensor.shape)

        connector._nccl_send = tracking_nccl_send  # type: ignore[assignment]

        await connector.send_free_seq(
            seq_ids=torch.tensor([10, 20], dtype=torch.int64)
        )

        assert len(nccl_calls) == 1
        assert nccl_calls[0] == (2,)


# ---------------------------------------------------------------------------
# recv_speculation metadata
# ---------------------------------------------------------------------------


class TestRecvSpeculation:
    @pytest.mark.anyio
    async def test_receives_speculation_response(self):
        connector = _make_connector()

        # Build a mock SpeculationResponse
        resp = SpeculationResponse(
            cache_hits_ref=TensorRef(
                shape=(2,), dtype="bool", buffer_id="ch", nbytes=2
            ),
            draft_tokens_ref=TensorRef(
                shape=(2, 5), dtype="int64", buffer_id="dt", nbytes=80
            ),
        )
        resp_bytes = encode(resp)

        async def mock_recv():
            return resp_bytes

        connector._zmq_recv = mock_recv  # type: ignore[assignment]

        recv_calls = []

        def tracking_nccl_recv(shape, dtype):
            t = torch.zeros(shape, dtype=dtype)
            recv_calls.append((shape, dtype))
            return t

        connector._nccl_recv = tracking_nccl_recv  # type: ignore[assignment]

        cache_hits, draft_tokens, draft_logits = (
            await connector.recv_speculation()
        )

        assert cache_hits.shape == (2,)
        assert draft_tokens.shape == (2, 5)
        assert draft_logits is None
        assert len(recv_calls) == 2

    @pytest.mark.anyio
    async def test_receives_logits_when_present(self):
        connector = _make_connector()

        resp = SpeculationResponse(
            cache_hits_ref=TensorRef(
                shape=(2,), dtype="bool", buffer_id="ch", nbytes=2
            ),
            draft_tokens_ref=TensorRef(
                shape=(2, 5), dtype="int64", buffer_id="dt", nbytes=80
            ),
            draft_logits_ref=TensorRef(
                shape=(2, 5, 100), dtype="float16", buffer_id="dl",
                nbytes=2000,
            ),
        )

        async def mock_recv():
            return encode(resp)

        connector._zmq_recv = mock_recv  # type: ignore[assignment]

        def tracking_nccl_recv(shape, dtype):
            return torch.zeros(shape, dtype=dtype)

        connector._nccl_recv = tracking_nccl_recv  # type: ignore[assignment]

        cache_hits, draft_tokens, draft_logits = (
            await connector.recv_speculation()
        )

        assert draft_logits is not None
        assert draft_logits.shape == (2, 5, 100)
