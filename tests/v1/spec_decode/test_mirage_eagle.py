# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Mirage EAGLE fused kernels module."""

import pytest


def test_import():
    """Module imports without mirage installed."""
    from vllm.v1.spec_decode.mirage_eagle import (
        MirageEagleFusedKernels,
        MirageEagleProposer,
        maybe_enable_mirage_eagle,
    )
    assert MirageEagleFusedKernels is not None
    assert MirageEagleProposer is not None
    assert maybe_enable_mirage_eagle is not None


def test_is_available():
    """is_available returns a bool regardless of mirage installation."""
    from vllm.v1.spec_decode.mirage_eagle import MirageEagleProposer
    result = MirageEagleProposer.is_available()
    assert isinstance(result, bool)


def test_config_field():
    """use_mirage_draft field exists in SpeculativeConfig."""
    try:
        from vllm.config.speculative import SpeculativeConfig
    except (ImportError, ModuleNotFoundError):
        pytest.skip("vllm.config requires CUDA platform")
    if hasattr(SpeculativeConfig, "model_fields"):
        assert "use_mirage_draft" in SpeculativeConfig.model_fields
        assert SpeculativeConfig.model_fields["use_mirage_draft"].default is False
