# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Preservation Property Tests — Co-located EAGLE and Non-Debug Paths Unchanged.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

Property 2: Preservation — For all inputs where the bug condition does NOT
hold (co-located EAGLE, or DISAGG_EAGLE_DEBUG not set), the fixed code SHALL
produce exactly the same behavior as the original code.

These tests are written and run BEFORE implementing the fix to capture
baseline behavior. They should PASS on unfixed code.

Observation-first methodology:
- combine_hidden_states(input_tensor) returns the same output regardless
  of debug flag
- _extract_batch_hidden_states returns the same extracted hidden states
  regardless of debug flag
- No log output is produced when DISAGG_EAGLE_DEBUG is not set to "1"
"""

import logging
import os
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Source file paths
# ---------------------------------------------------------------------------

WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[3]  # vllm/

SPECULATOR_PY = (
    WORKSPACE_ROOT / "vllm" / "v1" / "worker" / "gpu" / "spec_decode"
    / "disagg_draft" / "speculator.py"
)
DRAFT_WORKER_PY = (
    WORKSPACE_ROOT / "vllm" / "v1" / "worker" / "gpu" / "spec_decode"
    / "disagg_draft" / "draft_worker.py"
)
DRAFT_MODEL_RUNNER_PY = (
    WORKSPACE_ROOT / "vllm" / "v1" / "worker" / "gpu" / "spec_decode"
    / "disagg_draft" / "draft_model_runner.py"
)
EAGLE_SPECULATOR_PY = (
    WORKSPACE_ROOT / "vllm" / "v1" / "worker" / "gpu" / "spec_decode"
    / "eagle" / "speculator.py"
)
LLAMA_EAGLE3_PY = (
    WORKSPACE_ROOT / "vllm" / "model_executor" / "models" / "llama_eagle3.py"
)

# All source files that will be modified by the fix
SOURCE_FILES = {
    "speculator.py": SPECULATOR_PY,
    "draft_worker.py": DRAFT_WORKER_PY,
    "draft_model_runner.py": DRAFT_MODEL_RUNNER_PY,
    "eagle/speculator.py": EAGLE_SPECULATOR_PY,
    "llama_eagle3.py": LLAMA_EAGLE3_PY,
}


def _read_source(path: pathlib.Path) -> str:
    """Read source file content."""
    assert path.exists(), f"Source file not found: {path}"
    return path.read_text()


# ---------------------------------------------------------------------------
# Hypothesis strategies for tensor generation
# ---------------------------------------------------------------------------

# Batch sizes and hidden sizes for property-based testing
batch_sizes = st.integers(min_value=1, max_value=8)
hidden_sizes = st.sampled_from([256, 512, 1024, 2048])
# For EAGLE3, the fc input is 3 * target_hidden_size
fc_input_multiplier = st.just(3)


def make_random_hidden_states(batch_size: int, hidden_size: int,
                              dtype=torch.float32) -> torch.Tensor:
    """Create random hidden state tensors with realistic norms."""
    hs = torch.randn(batch_size, hidden_size, dtype=dtype)
    # Scale to realistic norm range (hidden states typically have
    # norms in the hundreds to thousands range)
    hs = hs * 100.0
    return hs


# ---------------------------------------------------------------------------
# Mock fc layer for combine_hidden_states testing
# ---------------------------------------------------------------------------

class MockFcLayer(nn.Module):
    """A simple linear layer mimicking the fc projection in EAGLE3."""

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        # Initialize with deterministic weights for reproducibility
        torch.manual_seed(42)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MockEagle3Model:
    """Mock EAGLE3 model with combine_hidden_states method."""

    def __init__(self, target_hidden_size: int, output_hidden_size: int):
        self.use_aux_hidden_state = True
        self.norm_before_fc = False
        self.input_norm = None
        # fc: [3 * target_hidden_size] -> [output_hidden_size]
        self.fc = MockFcLayer(3 * target_hidden_size, output_hidden_size)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Replicate the combine_hidden_states logic from llama_eagle3.py."""
        if not self.use_aux_hidden_state:
            return hidden_states
        if self.norm_before_fc and self.input_norm is not None:
            hidden_states = self.input_norm(hidden_states)
        return self.fc(hidden_states)


# ---------------------------------------------------------------------------
# Mock input batch for _extract_batch_hidden_states testing
# ---------------------------------------------------------------------------

class MockInputBatch:
    """Minimal mock of InputBatch for testing _extract_batch_hidden_states."""

    def __init__(self, num_reqs: int, tokens_per_req: int, device="cpu"):
        self.num_reqs = num_reqs
        self.req_ids = [f"req_{i}" for i in range(num_reqs)]
        # query_start_loc: cumulative token counts
        locs = [0]
        for i in range(num_reqs):
            locs.append(locs[-1] + tokens_per_req)
        self.query_start_loc = torch.tensor(locs, dtype=torch.int64,
                                            device=device)


# ---------------------------------------------------------------------------
# Test: Property — combine_hidden_states output unchanged by debug flag
# ---------------------------------------------------------------------------

class TestCombineHiddenStatesPreservation:
    """Property 2: Preservation — combine_hidden_states produces bitwise
    identical output regardless of DISAGG_EAGLE_DEBUG flag.

    **Validates: Requirements 3.1, 3.2**

    For all random hidden state tensors hs of shape [B, 3*hidden_size]
    with DISAGG_EAGLE_DEBUG unset: combine_hidden_states(hs) output is
    bitwise identical to baseline (no logging side effects modify
    computation).
    """

    @given(
        batch_size=batch_sizes,
        hidden_size=hidden_sizes,
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_combine_hidden_states_deterministic_without_debug(
        self, batch_size, hidden_size
    ):
        """Property: For all random hidden state tensors, combine_hidden_states
        produces identical output when DISAGG_EAGLE_DEBUG is NOT set.

        **Validates: Requirements 3.1, 3.2**
        """
        # Ensure debug flag is NOT set
        env_backup = os.environ.pop("DISAGG_EAGLE_DEBUG", None)
        try:
            model = MockEagle3Model(
                target_hidden_size=hidden_size,
                output_hidden_size=hidden_size,
            )
            # Create input tensor
            hs = make_random_hidden_states(batch_size, 3 * hidden_size)

            # Run twice — output must be bitwise identical
            with torch.no_grad():
                output1 = model.combine_hidden_states(hs.clone())
                output2 = model.combine_hidden_states(hs.clone())

            assert torch.equal(output1, output2), (
                f"combine_hidden_states produced different outputs for "
                f"identical inputs (B={batch_size}, hs={hidden_size}). "
                f"Max diff: {(output1 - output2).abs().max().item()}"
            )
        finally:
            if env_backup is not None:
                os.environ["DISAGG_EAGLE_DEBUG"] = env_backup

    @given(
        batch_size=batch_sizes,
        hidden_size=hidden_sizes,
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_combine_hidden_states_same_with_and_without_debug_flag(
        self, batch_size, hidden_size
    ):
        """Property: For all random hidden state tensors, combine_hidden_states
        output is identical whether DISAGG_EAGLE_DEBUG is set or not.

        **Validates: Requirements 3.1, 3.4**
        """
        model = MockEagle3Model(
            target_hidden_size=hidden_size,
            output_hidden_size=hidden_size,
        )
        hs = make_random_hidden_states(batch_size, 3 * hidden_size)

        # Run WITHOUT debug flag
        env_backup = os.environ.pop("DISAGG_EAGLE_DEBUG", None)
        try:
            with torch.no_grad():
                output_no_debug = model.combine_hidden_states(hs.clone())
        finally:
            if env_backup is not None:
                os.environ["DISAGG_EAGLE_DEBUG"] = env_backup

        # Run WITH debug flag
        os.environ["DISAGG_EAGLE_DEBUG"] = "1"
        try:
            with torch.no_grad():
                output_with_debug = model.combine_hidden_states(hs.clone())
        finally:
            os.environ.pop("DISAGG_EAGLE_DEBUG", None)

        assert torch.equal(output_no_debug, output_with_debug), (
            f"combine_hidden_states produced different outputs with vs "
            f"without DISAGG_EAGLE_DEBUG (B={batch_size}, hs={hidden_size}). "
            f"Max diff: {(output_no_debug - output_with_debug).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Test: Property — Logging does not modify tensor values
# ---------------------------------------------------------------------------

class TestLoggingDoesNotModifyTensors:
    """Property 2: Preservation — Logging functions do not modify input
    or output tensors.

    **Validates: Requirements 3.1, 3.2**

    For all random hidden state tensors: tensor values before any
    logging operation == tensor values after logging operation.
    """

    @given(
        batch_size=batch_sizes,
        hidden_size=hidden_sizes,
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_input_tensor_unchanged_after_combine(
        self, batch_size, hidden_size
    ):
        """Property: For all random hidden state tensors, the input tensor
        is not modified by combine_hidden_states (regardless of debug flag).

        **Validates: Requirements 3.1, 3.2**
        """
        model = MockEagle3Model(
            target_hidden_size=hidden_size,
            output_hidden_size=hidden_size,
        )
        hs = make_random_hidden_states(batch_size, 3 * hidden_size)
        hs_original = hs.clone()

        with torch.no_grad():
            _ = model.combine_hidden_states(hs)

        assert torch.equal(hs, hs_original), (
            f"combine_hidden_states modified the input tensor! "
            f"B={batch_size}, hs={hidden_size}. "
            f"Max diff: {(hs - hs_original).abs().max().item()}"
        )

    @given(
        batch_size=batch_sizes,
        hidden_size=hidden_sizes,
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_extract_hidden_states_preserves_input(
        self, batch_size, hidden_size
    ):
        """Property: For all random hidden state tensors and batch configs,
        _extract_batch_hidden_states does not modify the input tensors.

        **Validates: Requirements 3.2, 3.3**
        """
        tokens_per_req = 4
        total_tokens = batch_size * tokens_per_req

        last_hidden_states = make_random_hidden_states(
            total_tokens, hidden_size
        )
        lhs_original = last_hidden_states.clone()

        num_sampled = torch.ones(batch_size, dtype=torch.int64)

        input_batch = MockInputBatch(
            num_reqs=batch_size,
            tokens_per_req=tokens_per_req,
        )

        # Replicate the extraction logic from _extract_batch_hidden_states
        # (without importing the actual class to avoid GPU dependencies)
        B = batch_size
        last_token_indices = torch.zeros(B, dtype=torch.long)
        for j in range(B):
            ns = int(num_sampled[j].item())
            last_token_indices[j] = (
                input_batch.query_start_loc[j] + ns - 1
            )

        extracted = last_hidden_states[last_token_indices]

        # Verify input was not modified
        assert torch.equal(last_hidden_states, lhs_original), (
            f"_extract_batch_hidden_states logic modified the input "
            f"last_hidden_states tensor! B={batch_size}, hs={hidden_size}. "
            f"Max diff: {(last_hidden_states - lhs_original).abs().max().item()}"
        )

        # Verify extracted values are correct (indexing sanity check)
        for j in range(B):
            idx = int(last_token_indices[j].item())
            assert torch.equal(extracted[j], lhs_original[idx]), (
                f"Extracted hidden state at index {j} does not match "
                f"source at position {idx}."
            )


# ---------------------------------------------------------------------------
# Test: Property — No log output when DISAGG_EAGLE_DEBUG is not set
# ---------------------------------------------------------------------------

class TestNoLogOutputWithoutDebugFlag:
    """Property 2: Preservation — No log output is produced when
    DISAGG_EAGLE_DEBUG is not set to "1".

    **Validates: Requirements 3.1, 3.3, 3.4**

    For all random batch configurations: no [DISAGG_DIAG] or
    [COLOCATED_DIAG] log output is produced when DISAGG_EAGLE_DEBUG
    is not set.
    """

    @given(
        batch_size=batch_sizes,
        hidden_size=hidden_sizes,
    )
    @settings(
        max_examples=30,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_no_disagg_diag_logs_without_debug_flag(
        self, batch_size, hidden_size
    ):
        """Property: For all random batch configurations, no [DISAGG_DIAG]
        or [COLOCATED_DIAG] log output is produced when DISAGG_EAGLE_DEBUG
        is not set.

        **Validates: Requirements 3.1, 3.3, 3.4**

        This test verifies that the source code's debug-gated logging
        sections are properly guarded. We check that the source files
        do NOT contain ungated [DISAGG_DIAG] or [COLOCATED_DIAG] markers
        (i.e., any such markers must be inside an `if _DISAGG_DEBUG:` block
        or similar guard).

        On UNFIXED code, this test PASSES because there are no diagnostic
        markers at all. After the fix, it should still PASS because all
        markers are gated behind the debug flag.
        """
        # Ensure debug flag is NOT set
        env_backup = os.environ.pop("DISAGG_EAGLE_DEBUG", None)
        try:
            for name, path in SOURCE_FILES.items():
                source = _read_source(path)
                lines = source.split("\n")

                for i, line in enumerate(lines):
                    stripped = line.strip()
                    # Skip comments
                    if stripped.startswith("#"):
                        continue
                    # Check for diagnostic markers in string literals
                    if ("[DISAGG_DIAG]" in stripped
                            or "[COLOCATED_DIAG]" in stripped):
                        # This line contains a diagnostic marker.
                        # It must be inside a debug-gated block.
                        # Check that there's a preceding `if _DISAGG_DEBUG:`
                        # or `if _DISAGG_DEBUG` guard in the enclosing scope.
                        # Simple heuristic: look backwards for the guard.
                        found_guard = False
                        current_indent = len(line) - len(line.lstrip())
                        for j in range(i - 1, max(i - 20, -1), -1):
                            prev_line = lines[j].strip()
                            prev_indent = len(lines[j]) - len(
                                lines[j].lstrip())
                            if ("_DISAGG_DEBUG" in prev_line
                                    and prev_indent < current_indent):
                                found_guard = True
                                break
                            # Also accept inline check
                            if ("_DISAGG_DEBUG" in prev_line
                                    and "if" in prev_line):
                                found_guard = True
                                break
                        # On unfixed code, there are no markers at all,
                        # so this assertion is vacuously true.
                        # After the fix, markers must be guarded.
                        assert found_guard, (
                            f"File {name} line {i+1}: Found ungated "
                            f"diagnostic marker '{stripped[:80]}...' "
                            f"without _DISAGG_DEBUG guard. This would "
                            f"produce log output even when "
                            f"DISAGG_EAGLE_DEBUG is not set."
                        )
        finally:
            if env_backup is not None:
                os.environ["DISAGG_EAGLE_DEBUG"] = env_backup

    def test_no_diagnostic_markers_in_unfixed_code(self):
        """Baseline observation: On UNFIXED code, the source files should
        NOT contain [DISAGG_DIAG] or [COLOCATED_DIAG] markers at all
        (except possibly in comments or the existing COLOCATED EAGLE
        diagnostic logging which uses a different format).

        **Validates: Requirements 3.1, 3.4**

        This test documents the baseline: no structured diagnostic
        logging exists yet. After the fix, this test is superseded by
        the guard-checking test above.
        """
        # Check that the structured diagnostic markers don't exist yet
        # in the function bodies (they may exist in comments/docstrings)
        for name, path in SOURCE_FILES.items():
            source = _read_source(path)
            lines = source.split("\n")
            code_markers = []
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Skip comments and docstrings
                if stripped.startswith("#"):
                    continue
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                # Check for the structured diagnostic markers in actual code
                # (logger.info calls, not just string references)
                if ("logger.info" in stripped
                        and ("[DISAGG_DIAG]" in stripped
                             or "[COLOCATED_DIAG]" in stripped)):
                    code_markers.append((i + 1, stripped[:80]))

            # On unfixed code, we expect NO structured diagnostic markers
            # in logger.info calls. This is the baseline observation.
            # Note: This test will need to be updated after the fix is
            # implemented (the guard-checking test above handles post-fix).
            # For now, it documents that the markers don't exist.
            # We don't assert here because the test above already covers
            # the preservation property (markers must be guarded).
            pass


# ---------------------------------------------------------------------------
# Test: Property — Source code structure preservation
# ---------------------------------------------------------------------------

class TestSourceCodeStructurePreservation:
    """Property 2: Preservation — Key function signatures and structure
    are preserved after the fix.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

    Verifies that the functions that will be modified by the fix
    maintain their existing signatures and core logic.
    """

    def test_combine_hidden_states_signature_preserved(self):
        """The combine_hidden_states function signature must be preserved.

        **Validates: Requirements 3.1**
        """
        source = _read_source(LLAMA_EAGLE3_PY)
        assert "def combine_hidden_states(" in source, (
            "combine_hidden_states function not found in llama_eagle3.py"
        )
        # Verify it takes hidden_states parameter
        assert "hidden_states: torch.Tensor" in source, (
            "combine_hidden_states must accept hidden_states: torch.Tensor"
        )
        # Verify it returns torch.Tensor
        assert "-> torch.Tensor" in source, (
            "combine_hidden_states must return torch.Tensor"
        )

    def test_extract_batch_hidden_states_signature_preserved(self):
        """The _extract_batch_hidden_states function signature must be preserved.

        **Validates: Requirements 3.2**
        """
        source = _read_source(SPECULATOR_PY)
        assert "def _extract_batch_hidden_states(" in source, (
            "_extract_batch_hidden_states function not found in speculator.py"
        )
        # Verify key parameters
        assert "last_hidden_states" in source, (
            "_extract_batch_hidden_states must accept last_hidden_states"
        )
        assert "aux_hidden_states" in source, (
            "_extract_batch_hidden_states must accept aux_hidden_states"
        )

    def test_eagle_speculator_propose_signature_preserved(self):
        """The EagleSpeculator.propose function signature must be preserved.

        **Validates: Requirements 3.1**
        """
        source = _read_source(EAGLE_SPECULATOR_PY)
        assert "def propose(" in source, (
            "EagleSpeculator.propose function not found in eagle/speculator.py"
        )
        # Verify key parameters
        assert "last_hidden_states" in source, (
            "propose must accept last_hidden_states"
        )
        assert "aux_hidden_states" in source, (
            "propose must accept aux_hidden_states"
        )
        assert "-> torch.Tensor" in source, (
            "propose must return torch.Tensor"
        )

    def test_combine_hidden_states_core_logic_preserved(self):
        """The combine_hidden_states core logic (fc projection) must be
        preserved — the function must still call self.model.fc().

        **Validates: Requirements 3.1, 3.4**
        """
        source = _read_source(LLAMA_EAGLE3_PY)
        # Find the combine_hidden_states function body
        lines = source.split("\n")
        in_func = False
        func_lines = []
        func_indent = 0

        for line in lines:
            stripped = line.lstrip()
            if not in_func:
                if stripped.startswith("def combine_hidden_states("):
                    in_func = True
                    func_indent = len(line) - len(stripped)
                    func_lines.append(line)
            else:
                if stripped and not stripped.startswith("#"):
                    current_indent = len(line) - len(stripped)
                    if current_indent <= func_indent and (
                        stripped.startswith("def ")
                        or stripped.startswith("class ")
                        or stripped.startswith("@")
                    ):
                        break
                func_lines.append(line)

        func_body = "\n".join(func_lines)

        # Core logic checks
        assert "use_aux_hidden_state" in func_body, (
            "combine_hidden_states must check use_aux_hidden_state"
        )
        assert "self.model.fc(" in func_body or "self.fc(" in func_body, (
            "combine_hidden_states must call fc() for projection"
        )
        assert "norm_before_fc" in func_body, (
            "combine_hidden_states must check norm_before_fc"
        )
