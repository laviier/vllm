# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DraftServerMetrics on the draft server side."""

import importlib.util
import os
import sys
import types

import prometheus_client
import pytest


@pytest.fixture(autouse=True)
def _reset_prometheus_registry():
    """Reset the default Prometheus registry between tests to avoid
    duplicate metric registration errors."""
    collectors = list(prometheus_client.REGISTRY._names_to_collectors.values())
    for c in collectors:
        try:
            prometheus_client.REGISTRY.unregister(c)
        except Exception:
            pass
    yield


def _load_draft_server_module():
    """Load the draft_server module directly from file, bypassing
    vllm.__init__ which has a torch version incompatibility in this
    environment."""
    for pkg in [
        "vllm",
        "vllm.v1",
        "vllm.v1.spec_decode",
    ]:
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)

    # Stub vllm.config
    if "vllm.config" not in sys.modules:
        config_mod = types.ModuleType("vllm.config")
        config_mod.VllmConfig = type("VllmConfig", (), {})
        sys.modules["vllm.config"] = config_mod

    # Stub vllm.logger
    if "vllm.logger" not in sys.modules:
        import logging
        logger_mod = types.ModuleType("vllm.logger")
        logger_mod.init_logger = logging.getLogger
        sys.modules["vllm.logger"] = logger_mod

    # Stub draft_connector
    if "vllm.v1.spec_decode.draft_connector" not in sys.modules:
        dc_mod = types.ModuleType("vllm.v1.spec_decode.draft_connector")
        dc_mod._dtype_to_str = lambda d: str(d)
        dc_mod._str_to_dtype = lambda s: s
        sys.modules["vllm.v1.spec_decode.draft_connector"] = dc_mod

    # Stub draft_data_models
    if "vllm.v1.spec_decode.draft_data_models" not in sys.modules:
        ddm_mod = types.ModuleType("vllm.v1.spec_decode.draft_data_models")
        for name in [
            "DraftCommand", "FreeSeqRequest", "PrefillRequest",
            "SpeculationResponse", "TensorRef", "VerificationOutcome",
        ]:
            setattr(ddm_mod, name, type(name, (), {}))
        ddm_mod.decode = lambda *a, **kw: None
        ddm_mod.decode_command = lambda *a, **kw: None
        ddm_mod.encode = lambda *a, **kw: b""
        sys.modules["vllm.v1.spec_decode.draft_data_models"] = ddm_mod

    # Stub vllm.utils.network_utils
    if "vllm.utils" not in sys.modules:
        sys.modules["vllm.utils"] = types.ModuleType("vllm.utils")
    if "vllm.utils.network_utils" not in sys.modules:
        nu_mod = types.ModuleType("vllm.utils.network_utils")
        nu_mod.make_zmq_socket = lambda *a, **kw: None
        sys.modules["vllm.utils.network_utils"] = nu_mod

    mod_name = "vllm.v1.spec_decode.draft_server"
    # Force reload to pick up fresh prometheus registry
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    file_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..",
        "vllm", "v1", "spec_decode", "draft_server.py",
    )
    file_path = os.path.normpath(file_path)
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_metrics():
    mod = _load_draft_server_module()
    return mod.DraftServerMetrics()


class TestDraftServerMetrics:
    def test_initial_state(self):
        m = _make_metrics()
        assert m._total_lookups == 0
        assert m._total_hits == 0
        assert m.draft_batch_size._value.get() == 0
        assert m.draft_cache_hit_rate._value.get() == 0.0
        assert m.draft_connected_verify_servers._value.get() == 0
        assert m.draft_active_requests._value.get() == 0

    def test_batch_size_gauge(self):
        m = _make_metrics()
        m.draft_batch_size.set(8)
        assert m.draft_batch_size._value.get() == 8
        m.draft_batch_size.set(4)
        assert m.draft_batch_size._value.get() == 4

    def test_generation_latency_histogram(self):
        m = _make_metrics()
        m.draft_generation_latency.observe(0.035)
        assert m.draft_generation_latency._sum.get() == pytest.approx(0.035)
        m.draft_generation_latency.observe(0.015)
        assert m.draft_generation_latency._sum.get() == pytest.approx(0.05)

    def test_cache_hit_rate_rolling(self):
        m = _make_metrics()
        # 3 hits out of 10 lookups
        m._total_lookups += 10
        m._total_hits += 3
        m.draft_cache_hit_rate.set(m._total_hits / m._total_lookups)
        assert m.draft_cache_hit_rate._value.get() == pytest.approx(0.3)

        # 7 more hits out of 10 more lookups → 10/20 = 0.5
        m._total_lookups += 10
        m._total_hits += 7
        m.draft_cache_hit_rate.set(m._total_hits / m._total_lookups)
        assert m.draft_cache_hit_rate._value.get() == pytest.approx(0.5)

    def test_eviction_counter(self):
        m = _make_metrics()
        m.draft_eviction_count.inc(5)
        assert m.draft_eviction_count._value.get() == 5
        m.draft_eviction_count.inc(3)
        assert m.draft_eviction_count._value.get() == 8

    def test_connected_verify_servers_gauge(self):
        m = _make_metrics()
        m.draft_connected_verify_servers.set(3)
        assert m.draft_connected_verify_servers._value.get() == 3
        m.draft_connected_verify_servers.set(2)
        assert m.draft_connected_verify_servers._value.get() == 2

    def test_active_requests_gauge(self):
        m = _make_metrics()
        m.draft_active_requests.set(10)
        assert m.draft_active_requests._value.get() == 10
        m.draft_active_requests.set(7)
        assert m.draft_active_requests._value.get() == 7
