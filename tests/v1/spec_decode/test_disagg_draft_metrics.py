# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DisaggDraftMetrics on the verify server side."""

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


def _load_speculator_module():
    """Load the speculator module directly from file, bypassing
    vllm.__init__ which has a torch version incompatibility in this
    environment."""
    # Ensure parent packages exist as stubs so the loader doesn't
    # try to import vllm.__init__.
    for pkg in [
        "vllm",
        "vllm.v1",
        "vllm.v1.worker",
        "vllm.v1.worker.gpu",
        "vllm.v1.worker.gpu.spec_decode",
        "vllm.v1.worker.gpu.spec_decode.disagg_draft",
    ]:
        if pkg not in sys.modules:
            sys.modules[pkg] = types.ModuleType(pkg)

    # Stub vllm.config so `from vllm.config import VllmConfig` works
    if "vllm.config" not in sys.modules:
        config_mod = types.ModuleType("vllm.config")
        config_mod.VllmConfig = type("VllmConfig", (), {})  # type: ignore
        sys.modules["vllm.config"] = config_mod

    # Stub vllm.logger
    if "vllm.logger" not in sys.modules:
        import logging
        logger_mod = types.ModuleType("vllm.logger")
        logger_mod.init_logger = logging.getLogger  # type: ignore
        sys.modules["vllm.logger"] = logger_mod

    mod_name = "vllm.v1.worker.gpu.spec_decode.disagg_draft.speculator"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    file_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..",
        "vllm", "v1", "worker", "gpu", "spec_decode",
        "disagg_draft", "speculator.py",
    )
    file_path = os.path.normpath(file_path)
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_metrics():
    mod = _load_speculator_module()
    return mod.DisaggDraftMetrics()


class TestDisaggDraftMetrics:
    def test_initial_state(self):
        m = _make_metrics()
        assert m._total_requested == 0
        assert m._total_accepted == 0

    def test_record_increments_counters(self):
        m = _make_metrics()
        m.record_speculation(tokens_requested=10, tokens_accepted=7,
                             latency_s=0.05)
        assert m.draft_tokens_requested._value.get() == 10
        assert m.draft_tokens_accepted._value.get() == 7

    def test_record_updates_acceptance_rate_gauge(self):
        m = _make_metrics()
        m.record_speculation(tokens_requested=20, tokens_accepted=10,
                             latency_s=0.01)
        assert m.draft_acceptance_rate._value.get() == pytest.approx(0.5)

        m.record_speculation(tokens_requested=20, tokens_accepted=20,
                             latency_s=0.01)
        # cumulative: 30 accepted / 40 requested = 0.75
        assert m.draft_acceptance_rate._value.get() == pytest.approx(0.75)

    def test_record_observes_histogram(self):
        m = _make_metrics()
        m.record_speculation(tokens_requested=5, tokens_accepted=3,
                             latency_s=0.042)
        assert m.draft_round_trip_latency._sum.get() == pytest.approx(0.042)

    def test_zero_requested_no_division_error(self):
        m = _make_metrics()
        m.record_speculation(tokens_requested=0, tokens_accepted=0,
                             latency_s=0.01)
        assert m._total_requested == 0
        assert m.draft_acceptance_rate._value.get() == 0.0
