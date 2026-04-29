# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DisaggSpeculatorProxy graceful degradation (Task 5.2).

Validates Requirement 10.3: when the Verify_Server loses connection to
the Draft_Server, it continues serving without speculation.

NOTE: The speculator module has a deep import chain through vllm.config
which requires CUDA binaries.  We stub the heavy vllm.config import so
the speculator module can be loaded in a lightweight test environment.
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
import types
from collections import defaultdict
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch


# ------------------------------------------------------------------
# Bootstrap: make the speculator module importable without CUDA
# ------------------------------------------------------------------

def _bootstrap_speculator():
    """Load the speculator module with vllm.config stubbed out."""
    mod_name = (
        "vllm.v1.worker.gpu.spec_decode.disagg_draft.speculator"
    )
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    # Stub vllm.config with a mock that provides VllmConfig
    if "vllm.config" not in sys.modules:
        config_mock = types.ModuleType("vllm.config")
        config_mock.VllmConfig = MagicMock
        sys.modules["vllm.config"] = config_mock

    # Ensure intermediate packages exist
    for pkg_name in [
        "vllm.v1.worker.gpu.spec_decode",
        "vllm.v1.worker.gpu.spec_decode.disagg_draft",
    ]:
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = []
            pkg.__package__ = pkg_name
            sys.modules[pkg_name] = pkg

    # Load the speculator module from file
    spec_path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "vllm"
        / "v1"
        / "worker"
        / "gpu"
        / "spec_decode"
        / "disagg_draft"
        / "speculator.py"
    )
    spec = importlib.util.spec_from_file_location(mod_name, spec_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_speculator_mod = _bootstrap_speculator()
DisaggSpeculatorProxy = _speculator_mod.DisaggSpeculatorProxy

from vllm.v1.spec_decode.draft_router import DraftRouter


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_mock_connector(connected: bool = True):
    """Create a mock DraftConnector with a ``connected`` property."""
    conn = MagicMock(name="connector")
    type(conn).connected = PropertyMock(return_value=connected)
    return conn


def _make_router(n: int, available: list[bool] | None = None):
    """Build a DraftRouter with *n* mock connectors."""
    conns = [_make_mock_connector(connected=True) for _ in range(n)]
    router = DraftRouter(connectors=conns)
    if available is not None:
        for i, avail in enumerate(available):
            router._available[i] = avail
    return router


def _make_proxy(router=None, num_steps=3, max_reqs=4, vocab=32):
    """Build a minimal DisaggSpeculatorProxy with mocked config."""
    device = torch.device("cpu")

    # Minimal mock of VllmConfig / SpeculativeConfig
    spec_cfg = MagicMock()
    spec_cfg.num_speculative_tokens = num_steps
    spec_cfg.rejection_sample_method = "strict"
    spec_cfg.disagg_needs_hidden_states = False
    spec_cfg.method = "vanilla"
    spec_cfg.disagg_nccl_init_method = None

    draft_model_cfg = MagicMock()
    draft_model_cfg.get_vocab_size.return_value = vocab

    spec_cfg.draft_model_config = draft_model_cfg

    model_cfg = MagicMock()
    model_cfg.dtype = torch.float32
    model_cfg.get_hidden_size.return_value = 64
    model_cfg.get_vocab_size.return_value = vocab

    sched_cfg = MagicMock()
    sched_cfg.max_num_seqs = max_reqs

    cache_cfg = MagicMock()
    cache_cfg.enable_prefix_caching = False

    vllm_cfg = MagicMock()
    vllm_cfg.speculative_config = spec_cfg
    vllm_cfg.model_config = model_cfg
    vllm_cfg.scheduler_config = sched_cfg
    vllm_cfg.cache_config = cache_cfg

    # get_tp_group is imported inside __init__'s try/except, so it
    # will gracefully fall back to _tp_rank=0 without patching.
    proxy = DisaggSpeculatorProxy(vllm_cfg, device)

    if router is not None:
        proxy.set_router(router)

    return proxy


def _make_input_batch(req_ids: list[str], device=torch.device("cpu")):
    """Create a minimal mock input_batch."""
    n = len(req_ids)
    batch = MagicMock()
    batch.num_reqs = n
    batch.req_ids = req_ids
    batch.idx_mapping = torch.arange(n, dtype=torch.int64, device=device)
    # query_start_loc: simple 1-token-per-request layout
    batch.query_start_loc = torch.arange(
        n + 1, dtype=torch.int64, device=device
    )
    batch.input_ids = torch.zeros(n, dtype=torch.int64, device=device)
    return batch


# ------------------------------------------------------------------
# Tests: propose() returns zero tensor when all servers unavailable
# ------------------------------------------------------------------


class TestAllServersUnavailable:
    """When all connectors are unavailable, propose() returns zeros."""

    def test_returns_zero_tensor(self):
        router = _make_router(2, available=[False, False])
        proxy = _make_proxy(router=router, num_steps=3)

        batch = _make_input_batch(["r0", "r1"])
        num_sampled = torch.tensor([1, 1])
        num_rejected = torch.tensor([0, 0])
        last_sampled = torch.tensor([[5], [6]])
        temperature = torch.tensor([1.0, 1.0])
        seeds = torch.tensor([0, 0])
        hs = torch.zeros(2, 64)

        result = proxy.propose(
            input_batch=batch,
            attn_metadata={},
            slot_mappings={},
            last_hidden_states=hs,
            aux_hidden_states=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            last_sampled=last_sampled,
            next_prefill_tokens=torch.tensor([]),
            temperature=temperature,
            seeds=seeds,
        )

        assert result.shape == (2, 3)
        assert (result == 0).all()

    def test_returns_zero_tensor_single_server(self):
        router = _make_router(1, available=[False])
        proxy = _make_proxy(router=router, num_steps=5)

        batch = _make_input_batch(["r0"])
        result = proxy.propose(
            input_batch=batch,
            attn_metadata={},
            slot_mappings={},
            last_hidden_states=torch.zeros(1, 64),
            aux_hidden_states=None,
            num_sampled=torch.tensor([1]),
            num_rejected=torch.tensor([0]),
            last_sampled=torch.tensor([[7]]),
            next_prefill_tokens=torch.tensor([]),
            temperature=torch.tensor([1.0]),
            seeds=torch.tensor([0]),
        )

        assert result.shape == (1, 5)
        assert (result == 0).all()


# ------------------------------------------------------------------
# Tests: ConnectionError in _do_propose_nm marks server unavailable
# ------------------------------------------------------------------


class TestConnectionErrorHandling:
    """ConnectionError during speculation marks server unavailable."""

    def test_connection_error_marks_server_failed(self):
        router = _make_router(2, available=[True, True])
        proxy = _make_proxy(router=router, num_steps=3)

        # Simulate: server 0 raises ConnectionError on send
        async def _raise_conn_error(*args, **kwargs):
            raise ConnectionError("server down")

        router.connectors[0].send_verification_outcome = _raise_conn_error

        # Assign a request to server 0
        router.assign("r0")
        assert router.assignment["r0"] == 0

        # Prefill the request so it has a seq_id
        proxy._disagg_req_to_seq_id["r0"] = 0
        proxy._disagg_prefilled_reqs.add("r0")

        # Call _do_propose_nm — should catch ConnectionError and
        # call handle_server_failure
        draft_toks, draft_logits = proxy._do_propose_nm(
            active_req_ids=["r0"],
            active_req_indices=[0],
            seq_ids=torch.tensor([0], dtype=torch.int64),
            k_accepted=torch.tensor([0], dtype=torch.int64),
            bonus_tokens=torch.tensor([5], dtype=torch.int64),
            temperatures=torch.tensor([1.0]),
            hidden_states=None,
            extend_counts=None,
            extend_hidden_states=None,
            extend_token_ids=None,
            B_active=1,
        )

        # Server 0 should now be marked unavailable
        assert router._available[0] is False
        assert router._available[1] is True

        # Draft tokens should be zeros (fallback)
        assert (draft_toks == 0).all()


# ------------------------------------------------------------------
# Tests: Reconnection attempts
# ------------------------------------------------------------------


class TestReconnection:
    """Periodically attempt to reconnect unavailable servers."""

    def test_reconnect_restores_server(self):
        router = _make_router(2, available=[False, True])
        proxy = _make_proxy(router=router)

        # Connector 0 is disconnected, _reconnect will succeed
        conn0 = router.connectors[0]
        connected_values = iter([False, True])
        type(conn0).connected = PropertyMock(
            side_effect=lambda: next(connected_values)
        )
        conn0._reconnect = MagicMock()

        proxy._attempt_reconnect_unavailable_servers()

        conn0._reconnect.assert_called_once()
        assert router._available[0] is True

    def test_reconnect_fails_keeps_unavailable(self):
        router = _make_router(2, available=[False, True])
        proxy = _make_proxy(router=router)

        # Connector 0 stays disconnected after _reconnect
        conn0 = router.connectors[0]
        type(conn0).connected = PropertyMock(return_value=False)
        conn0._reconnect = MagicMock()

        proxy._attempt_reconnect_unavailable_servers()

        conn0._reconnect.assert_called_once()
        assert router._available[0] is False

    def test_reconnect_only_called_periodically(self):
        router = _make_router(1, available=[False])
        proxy = _make_proxy(router=router)
        proxy._reconnect_check_interval = 5

        # Connector stays disconnected
        conn = router.connectors[0]
        type(conn).connected = PropertyMock(return_value=False)
        conn._reconnect = MagicMock()

        batch = _make_input_batch(["r0"])
        kwargs = dict(
            input_batch=batch,
            attn_metadata={},
            slot_mappings={},
            last_hidden_states=torch.zeros(1, 64),
            aux_hidden_states=None,
            num_sampled=torch.tensor([1]),
            num_rejected=torch.tensor([0]),
            last_sampled=torch.tensor([[7]]),
            next_prefill_tokens=torch.tensor([]),
            temperature=torch.tensor([1.0]),
            seeds=torch.tensor([0]),
        )

        # Call propose multiple times
        for _ in range(12):
            proxy.propose(**kwargs)

        # _reconnect should be called at propose_count 5 and 10
        # (every _reconnect_check_interval calls)
        assert conn._reconnect.call_count == 2

    def test_no_reconnect_when_all_available(self):
        router = _make_router(2, available=[True, True])
        proxy = _make_proxy(router=router)

        conn0 = router.connectors[0]
        conn0._reconnect = MagicMock()
        conn1 = router.connectors[1]
        conn1._reconnect = MagicMock()

        proxy._attempt_reconnect_unavailable_servers()

        conn0._reconnect.assert_not_called()
        conn1._reconnect.assert_not_called()
