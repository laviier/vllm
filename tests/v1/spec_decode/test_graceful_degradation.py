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
from unittest.mock import AsyncMock, MagicMock, PropertyMock

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
    conn.send_prefill = AsyncMock()
    conn.send_free_seq = AsyncMock()
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
    spec_cfg.method = "vanilla"

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
# Tests: old and new model-runner propose interfaces
# ------------------------------------------------------------------


class TestProposeInterface:
    def test_exposes_new_model_runner_lifecycle_hooks(self):
        proxy = _make_proxy(router=_make_router(1))

        assert isinstance(proxy.model, torch.nn.Module)
        proxy.init_cudagraph_manager()
        proxy.capture()

    def test_accepts_new_model_runner_positional_signature(self):
        proxy = _make_proxy(router=_make_router(1))
        batch = _make_input_batch(["r0"])
        num_sampled = torch.tensor([1])
        last_sampled = torch.tensor([[7]])
        temperature = torch.tensor([1.0])
        expected = torch.tensor([[7, 8, 9]])
        proxy._do_propose = MagicMock(return_value=expected)

        result = proxy.propose(
            batch,
            {},
            {},
            torch.zeros(1, 64),
            None,
            num_sampled,
            torch.tensor([0]),
            last_sampled,
            torch.tensor([]),
            temperature,
            torch.tensor([0]),
        )

        assert result is expected
        proxy._do_propose.assert_called_once_with(
            batch, num_sampled, last_sampled, temperature
        )

    def test_preserves_legacy_positional_signature(self):
        proxy = _make_proxy(router=_make_router(1))
        batch = _make_input_batch(["r0"])
        num_sampled = torch.tensor([1])
        last_sampled = torch.tensor([[7]])
        temperature = torch.tensor([1.0])
        expected = torch.tensor([[7, 8, 9]])
        proxy._do_propose = MagicMock(return_value=expected)

        result = proxy.propose(
            batch,
            num_sampled,
            last_sampled,
            temperature,
        )

        assert result is expected
        proxy._do_propose.assert_called_once_with(
            batch, num_sampled, last_sampled, temperature
        )

    def test_dummy_run_skips_remote_speculation(self):
        proxy = _make_proxy(router=_make_router(1))
        batch = _make_input_batch(["r0"])
        proxy._do_propose = MagicMock()

        result = proxy.propose(
            input_batch=batch,
            num_sampled=torch.tensor([1]),
            last_sampled=torch.tensor([[7]]),
            temperature=torch.tensor([1.0]),
            dummy_run=True,
        )

        assert result.shape == (1, 3)
        assert (result == 0).all()
        proxy._do_propose.assert_not_called()


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
# Tests: PREFILL state and failover
# ------------------------------------------------------------------


class TestPrefillState:
    """Only successfully prefilled requests may speculate."""

    def test_missing_prompt_does_not_mark_request_prefilled(self):
        router = _make_router(1)
        proxy = _make_proxy(router=router, num_steps=3)

        batch = _make_input_batch(["r0"])
        ctx = proxy._do_propose_dispatch(
            input_batch=batch,
            num_sampled=torch.tensor([1]),
            last_sampled=torch.tensor([[5]]),
            temperature=torch.tensor([1.0]),
        )

        assert ctx is None
        assert "r0" in proxy._disagg_req_to_seq_id
        assert "r0" not in proxy._disagg_prefilled_reqs
        router.connectors[0].send_prefill.assert_not_awaited()

    def test_failed_prefill_keeps_prompt_for_retry(self):
        router = _make_router(1)
        proxy = _make_proxy(router=router)
        connector = router.connectors[0]
        connector.send_prefill.side_effect = [
            RuntimeError("temporary send failure"),
            None,
        ]
        proxy.cache_new_request_tokens("r0", [1, 2, 3])

        proxy._prefill_new_requests(["r0"])

        assert "r0" not in proxy._disagg_prefilled_reqs
        assert proxy._pending_prompt_tokens["r0"] == [1, 2, 3]

        proxy._prefill_new_requests(["r0"])

        assert "r0" in proxy._disagg_prefilled_reqs
        assert connector.send_prefill.await_count == 2

    def test_connection_failure_invalidates_same_server_prefills(self):
        router = _make_router(2)
        proxy = _make_proxy(router=router)
        connector = router.connectors[0]
        connector.send_prefill.side_effect = [
            None,
            ConnectionError("server failed"),
        ]
        router.assignment.update({"r0": 0, "r1": 0})
        proxy.cache_new_request_tokens("r0", [1])
        proxy.cache_new_request_tokens("r1", [2])

        proxy._prefill_new_requests(["r0", "r1"])

        assert proxy._disagg_prefilled_reqs.isdisjoint({"r0", "r1"})
        assert router.assignment["r0"] == 1
        assert router.assignment["r1"] == 1


class TestServerFailover:
    """Reassigned requests replay PREFILL before speculation resumes."""

    def test_failover_invalidates_and_reprefills_request(self):
        router = _make_router(2)
        proxy = _make_proxy(router=router)
        proxy.cache_new_request_tokens("r0", [4, 5, 6])
        proxy._disagg_req_to_seq_id["r0"] = 7
        proxy._disagg_prefilled_reqs.add("r0")
        router.assign("r0")
        assert router.assignment["r0"] == 0

        proxy._handle_server_failure(0)

        assert router.assignment["r0"] == 1
        assert "r0" not in proxy._disagg_prefilled_reqs

        proxy._prefill_new_requests(["r0"])

        assert "r0" in proxy._disagg_prefilled_reqs
        router.connectors[1].send_prefill.assert_awaited_once()
        call = router.connectors[1].send_prefill.await_args
        assert call.kwargs["seq_id"] == 7
        assert call.kwargs["prompt_token_ids"].tolist() == [4, 5, 6]


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
