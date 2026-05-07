# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DraftServer request-namespacing and cleanup helpers.

Handler-level behavior (prefill, speculate, free_seq, eviction) is
exercised by integration benchmarks rather than mocked unit tests —
the handlers mutate GPU state and depend on a loaded draft model, so
mocking them here produced mostly tautological tests that decayed
whenever the wire protocol evolved.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vllm.v1.spec_decode.draft_server import DraftServer


def _make_vllm_config() -> MagicMock:
    cfg = MagicMock()
    cfg.speculative_config = MagicMock()
    cfg.speculative_config.num_speculative_tokens = 5
    cfg.speculative_config.disagg_fan_out = 1
    cfg.speculative_config.disagg_saguaro_c = None
    cfg.speculative_config.disagg_jit_fallback = True
    cfg.scheduler_config = MagicMock()
    cfg.scheduler_config.max_num_seqs = 32
    cfg.model_config = MagicMock()
    cfg.model_config.dtype = "float16"
    cfg.model_config.get_vocab_size.return_value = 32000
    cfg.speculative_config.draft_model_config = MagicMock()
    cfg.speculative_config.draft_model_config.get_vocab_size.return_value = (
        32000
    )
    return cfg


@pytest.fixture
def draft_server(tmp_path):
    addr = f"ipc://{tmp_path}/draft_server_test"
    server = DraftServer(
        vllm_config=_make_vllm_config(),
        bind_address=addr,
    )
    yield server
    server._cleanup()


class TestRequestNamespacing:
    """Verify composite key (verify_server_id, seq_id) namespacing."""

    def test_make_key(self, draft_server: DraftServer):
        assert draft_server._make_key("vs-1", 42) == ("vs-1", 42)

    def test_register_request_creates_state(self, draft_server: DraftServer):
        key = draft_server._register_request("vs-1", 10)
        assert key in draft_server._request_state
        assert key in draft_server._verify_servers["vs-1"]

    def test_register_same_seq_id_different_servers(
        self, draft_server: DraftServer
    ):
        k1 = draft_server._register_request("vs-1", 1)
        k2 = draft_server._register_request("vs-2", 1)
        assert k1 != k2
        assert k1 in draft_server._request_state
        assert k2 in draft_server._request_state

    def test_unregister_request_removes_state(
        self, draft_server: DraftServer
    ):
        draft_server._register_request("vs-1", 5)
        draft_server._unregister_request("vs-1", 5)
        assert ("vs-1", 5) not in draft_server._request_state
        assert "vs-1" not in draft_server._verify_servers

    def test_unregister_nonexistent_is_safe(
        self, draft_server: DraftServer
    ):
        draft_server._unregister_request("vs-99", 999)

    def test_get_request_state_creates_if_absent(
        self, draft_server: DraftServer
    ):
        key = ("vs-1", 7)
        state = draft_server._get_request_state(key)
        assert isinstance(state, dict)
        assert key in draft_server._request_state

    def test_multiple_requests_per_server(
        self, draft_server: DraftServer
    ):
        draft_server._register_request("vs-1", 1)
        draft_server._register_request("vs-1", 2)
        draft_server._register_request("vs-1", 3)
        assert len(draft_server._verify_servers["vs-1"]) == 3

        draft_server._unregister_request("vs-1", 2)
        assert len(draft_server._verify_servers["vs-1"]) == 2
        assert ("vs-1", 2) not in draft_server._request_state


class TestCleanup:
    def test_cleanup_clears_state(self, draft_server: DraftServer):
        draft_server._register_request("vs-1", 1)
        draft_server._register_request("vs-2", 2)
        draft_server._cleanup()
        assert not draft_server._request_state
        assert not draft_server._verify_servers

    def test_double_cleanup_is_safe(self, draft_server: DraftServer):
        draft_server._cleanup()
        draft_server._cleanup()


class TestTimeoutEviction:
    def test_eviction_timeout_default(self, draft_server: DraftServer):
        assert draft_server._eviction_timeout_s > 0

    def test_last_seen_initialized_empty(self, draft_server: DraftServer):
        assert draft_server._verify_server_last_seen == {}
