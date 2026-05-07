# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for validate_draft_server_connectivity startup validation."""

import threading

import pytest
import zmq

from vllm.v1.spec_decode.draft_connector import (
    validate_draft_server_connectivity,
)
from vllm.v1.spec_decode.draft_data_models import (
    decode_command,
    encode_command,
)


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_mock_draft_server(addr: str, stop_event: threading.Event) -> None:
    """Run a minimal ZMQ REP server that responds to HEALTHCHECK."""
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(addr)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    while not stop_event.is_set():
        events = dict(poller.poll(timeout=100))
        if sock in events:
            msg = sock.recv()
            # Echo back a HEALTHCHECK response
            reply = encode_command("HEALTHCHECK")
            sock.send(reply)

    sock.close(linger=0)
    ctx.term()


class TestValidateDraftServerConnectivity:
    """Tests for the startup validation function."""

    def test_empty_addresses_raises(self):
        """Empty address list should raise immediately."""
        with pytest.raises(RuntimeError, match="empty"):
            validate_draft_server_connectivity([])

    def test_unreachable_server_raises(self):
        """When no server is listening, should raise RuntimeError."""
        port = _find_free_port()
        addr = f"tcp://127.0.0.1:{port}"
        with pytest.raises(RuntimeError, match="No draft servers reachable"):
            validate_draft_server_connectivity([addr], timeout_ms=500)

    def test_reachable_server_passes(self):
        """When a server responds to HEALTHCHECK, validation passes."""
        port = _find_free_port()
        addr = f"tcp://127.0.0.1:{port}"

        stop_event = threading.Event()
        server_thread = threading.Thread(
            target=_run_mock_draft_server,
            args=(addr, stop_event),
            daemon=True,
        )
        server_thread.start()

        try:
            # Should not raise
            validate_draft_server_connectivity([addr], timeout_ms=3000)
        finally:
            stop_event.set()
            server_thread.join(timeout=5)

    def test_mixed_reachable_and_unreachable(self):
        """If at least one server is reachable, validation passes."""
        good_port = _find_free_port()
        bad_port = _find_free_port()
        good_addr = f"tcp://127.0.0.1:{good_port}"
        bad_addr = f"tcp://127.0.0.1:{bad_port}"

        stop_event = threading.Event()
        server_thread = threading.Thread(
            target=_run_mock_draft_server,
            args=(good_addr, stop_event),
            daemon=True,
        )
        server_thread.start()

        try:
            # Should not raise — one server is reachable
            validate_draft_server_connectivity(
                [bad_addr, good_addr], timeout_ms=1000
            )
        finally:
            stop_event.set()
            server_thread.join(timeout=5)

    def test_all_unreachable_raises_with_addresses(self):
        """Error message should list the addresses that were tried."""
        port1 = _find_free_port()
        port2 = _find_free_port()
        addr1 = f"tcp://127.0.0.1:{port1}"
        addr2 = f"tcp://127.0.0.1:{port2}"

        with pytest.raises(RuntimeError, match="2 address"):
            validate_draft_server_connectivity(
                [addr1, addr2], timeout_ms=500
            )

    def test_server_sends_valid_healthcheck_response(self):
        """Verify the mock server's HEALTHCHECK response is parseable."""
        port = _find_free_port()
        addr = f"tcp://127.0.0.1:{port}"

        stop_event = threading.Event()
        server_thread = threading.Thread(
            target=_run_mock_draft_server,
            args=(addr, stop_event),
            daemon=True,
        )
        server_thread.start()

        try:
            # Manually connect and verify the response
            ctx = zmq.Context()
            sock = ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.SNDTIMEO, 3000)
            sock.setsockopt(zmq.RCVTIMEO, 3000)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(addr)

            healthcheck_bytes = encode_command("HEALTHCHECK")
            sock.send(healthcheck_bytes)
            reply = sock.recv()

            cmd = decode_command(reply)
            assert cmd.command == "HEALTHCHECK"
            assert cmd.payload == b""

            sock.close(linger=0)
            ctx.term()
        finally:
            stop_event.set()
            server_thread.join(timeout=5)
