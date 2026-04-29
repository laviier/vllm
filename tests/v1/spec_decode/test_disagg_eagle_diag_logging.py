# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Bug Condition Exploration Test — Diagnostic Logging Absent on Unfixed Code.

**Validates: Requirements 1.3, 1.4, 1.5, 2.2, 2.3, 2.4, 2.5**

Property 1: Bug Condition — For all inputs where isBugCondition(input) holds
(disagg EAGLE active, NCCL transfer, debug enabled via DISAGG_EAGLE_DEBUG=1),
the pipeline checkpoint functions SHALL produce structured log output with
the [DISAGG_DIAG] or [COLOCATED_DIAG] prefix.

This test is EXPECTED TO FAIL on unfixed code — failure confirms the bug
(missing diagnostic logging) exists. DO NOT fix the test or the code when
it fails.

Approach: We use a scoped PBT strategy that generates random checkpoint
configurations and verifies that the source code for each checkpoint
function contains the required diagnostic logging markers. This directly
surfaces the bug condition: the absence of structured logging at pipeline
checkpoints.
"""

import os
import pathlib

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Source file paths (relative to workspace root)
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


def _read_source(path: pathlib.Path) -> str:
    """Read source file content."""
    assert path.exists(), f"Source file not found: {path}"
    return path.read_text()


def _read_function_source(file_path: pathlib.Path, func_name: str) -> str:
    """Extract the source of a specific function/method from a file.

    Uses a simple indentation-based parser to find the function body.
    Returns the full function source including the def line.
    """
    source = _read_source(file_path)
    lines = source.split("\n")
    in_func = False
    func_lines = []
    func_indent = 0

    for line in lines:
        stripped = line.lstrip()
        if not in_func:
            if stripped.startswith(f"def {func_name}(") or \
               stripped.startswith(f"def {func_name} ("):
                in_func = True
                func_indent = len(line) - len(stripped)
                func_lines.append(line)
        else:
            # Check if we've left the function (non-empty line at same
            # or lower indent level that starts a new def/class)
            if stripped and not stripped.startswith("#"):
                current_indent = len(line) - len(stripped)
                if current_indent <= func_indent and (
                    stripped.startswith("def ")
                    or stripped.startswith("class ")
                    or stripped.startswith("@")
                ):
                    break
            func_lines.append(line)

    return "\n".join(func_lines)


# ---------------------------------------------------------------------------
# Checkpoint definitions for PBT
# ---------------------------------------------------------------------------

CHECKPOINTS = {
    "CP1": {
        "name": "Hidden State Extraction",
        "file": SPECULATOR_PY,
        "function": "_extract_batch_hidden_states",
        "marker": "[DISAGG_DIAG][CP1]",
        "description": (
            "Extraction indices, norms, dtype, first values per request"
        ),
        "requirements": "1.3, 2.3",
    },
    "CP2": {
        "name": "NCCL Receive Per-Request Norms",
        "file": DRAFT_WORKER_PY,
        "function": "_handle_speculation",
        "marker": "[DISAGG_DIAG][CP2]",
        "description": (
            "Per-request hidden state norms after NCCL receive"
        ),
        "requirements": "1.3, 2.2",
    },
    "CP3": {
        "name": "Glue Decode Inputs",
        "file": DRAFT_WORKER_PY,
        "function": "_run_glue_decode",
        "marker": "[DISAGG_DIAG][CP3]",
        "description": (
            "fused_ids, positions, per-token hs norms before fc"
        ),
        "requirements": "1.4, 2.4",
    },
    "CP4": {
        "name": "Post-fc Projection",
        "file": DRAFT_WORKER_PY,
        "function": "_run_glue_decode",
        "marker": "[DISAGG_DIAG][CP4]",
        "description": (
            "Per-token norms after fc projection, pre vs post comparison"
        ),
        "requirements": "1.4, 2.4",
    },
    "CP5": {
        "name": "Glue Decode Outputs",
        "file": DRAFT_WORKER_PY,
        "function": "_run_glue_decode",
        "marker": "[DISAGG_DIAG][CP5]",
        "description": (
            "Prenorm norms per request, logits top-5 at recovery position"
        ),
        "requirements": "1.4, 2.4",
    },
    "CP7": {
        "name": "Per-Step JIT Speculation",
        "file": DRAFT_MODEL_RUNNER_PY,
        "function": "eagle_sequential_speculate",
        "marker": "[DISAGG_DIAG][CP7]",
        "description": (
            "Per-step input hs norm, output prenorm norm, top-5 logits, "
            "sampled token, position"
        ),
        "requirements": "1.5, 2.5",
    },
    "CP8": {
        "name": "Co-located Comparison Logs",
        "file": EAGLE_SPECULATOR_PY,
        "function": "propose",
        "marker": "[COLOCATED_DIAG][CP8]",
        "description": (
            "Raw hs norms, fc-projected norms, per-step hs norms, "
            "top-5 logits, sampled tokens"
        ),
        "requirements": "2.2",
    },
    "CP9": {
        "name": "fc Projection Internals",
        "file": LLAMA_EAGLE3_PY,
        "function": "combine_hidden_states",
        "marker": "[DISAGG_DIAG][CP9]",
        "description": (
            "Input norm, output norm, norm_before_fc flag, shape, dtype"
        ),
        "requirements": "2.2",
    },
}

# Strategy: pick a random checkpoint to test
checkpoint_ids = st.sampled_from(list(CHECKPOINTS.keys()))


# ---------------------------------------------------------------------------
# Debug gate check: DISAGG_EAGLE_DEBUG module-level flag
# ---------------------------------------------------------------------------

DEBUG_GATE_FILES = {
    "speculator.py": SPECULATOR_PY,
    "draft_worker.py": DRAFT_WORKER_PY,
    "draft_model_runner.py": DRAFT_MODEL_RUNNER_PY,
    "eagle/speculator.py": EAGLE_SPECULATOR_PY,
    "llama_eagle3.py": LLAMA_EAGLE3_PY,
}

debug_gate_file_ids = st.sampled_from(list(DEBUG_GATE_FILES.keys()))


# ---------------------------------------------------------------------------
# Test: Property — Diagnostic logging markers present at all checkpoints
# ---------------------------------------------------------------------------

class TestBugConditionDiagnosticLogging:
    """Property 1: Bug Condition — Diagnostic Logging Absent on Unfixed Code.

    For all checkpoint configurations where isBugCondition(input) holds,
    the source code SHALL contain structured diagnostic logging markers.

    This test FAILS on unfixed code, confirming the bug exists.
    """

    @given(cp_id=checkpoint_ids)
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_checkpoint_has_diagnostic_marker(self, cp_id):
        """Property: For all checkpoints, the function source SHALL contain
        the structured [DISAGG_DIAG] or [COLOCATED_DIAG] marker.

        **Validates: Requirements 1.3, 1.4, 1.5, 2.2, 2.3, 2.4, 2.5**
        """
        cp = CHECKPOINTS[cp_id]
        func_source = _read_function_source(cp["file"], cp["function"])

        assert func_source, (
            f"Could not find function {cp['function']} in {cp['file']}"
        )

        marker = cp["marker"]
        has_marker = marker in func_source

        assert has_marker, (
            f"Checkpoint {cp_id} ({cp['name']}): "
            f"Function {cp['function']} in {cp['file'].name} "
            f"does NOT contain '{marker}' diagnostic logging. "
            f"Missing: {cp['description']}. "
            f"Requirements: {cp['requirements']}"
        )

    @given(file_id=debug_gate_file_ids)
    @settings(
        max_examples=10,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_debug_gate_present(self, file_id):
        """Property: For all source files in the pipeline, a module-level
        debug gate `_DISAGG_DEBUG = os.environ.get("DISAGG_EAGLE_DEBUG"...`
        SHALL be present to control diagnostic logging.

        **Validates: Requirements 2.2, 2.3, 2.4, 2.5**
        """
        file_path = DEBUG_GATE_FILES[file_id]
        source = _read_source(file_path)

        has_debug_gate = (
            "_DISAGG_DEBUG" in source
            and "DISAGG_EAGLE_DEBUG" in source
        )

        assert has_debug_gate, (
            f"File {file_path.name} does NOT contain the "
            f"_DISAGG_DEBUG module-level debug gate. "
            f"Diagnostic logging cannot be controlled by "
            f"DISAGG_EAGLE_DEBUG=1 environment variable."
        )


# ---------------------------------------------------------------------------
# Test: Specific checkpoint coverage (non-PBT, for clear counterexamples)
# ---------------------------------------------------------------------------

class TestCheckpointCoverage:
    """Exhaustive checkpoint coverage tests for clear counterexample
    documentation. Each test checks one specific checkpoint."""

    def test_cp1_extraction_logging(self):
        """Checkpoint 1: _extract_batch_hidden_states SHALL contain
        [DISAGG_DIAG][CP1] logging for extraction indices, norms, dtype.

        **Validates: Requirements 1.3, 2.3**
        """
        source = _read_function_source(SPECULATOR_PY,
                                       "_extract_batch_hidden_states")
        assert "[DISAGG_DIAG][CP1]" in source, (
            "CP1 MISSING: _extract_batch_hidden_states has no "
            "[DISAGG_DIAG][CP1] diagnostic logging for extraction "
            "indices, norms, dtype, first values."
        )

    def test_cp2_nccl_receive_logging(self):
        """Checkpoint 2: _handle_speculation SHALL contain
        [DISAGG_DIAG][CP2] logging for per-request NCCL receive norms.

        **Validates: Requirements 1.3, 2.2**
        """
        source = _read_function_source(DRAFT_WORKER_PY,
                                       "_handle_speculation")
        assert "[DISAGG_DIAG][CP2]" in source, (
            "CP2 MISSING: _handle_speculation has no "
            "[DISAGG_DIAG][CP2] diagnostic logging for per-request "
            "hidden state norms after NCCL receive."
        )

    def test_cp3_glue_decode_inputs_logging(self):
        """Checkpoint 3: _run_glue_decode SHALL contain
        [DISAGG_DIAG][CP3] logging for glue decode inputs.

        **Validates: Requirements 1.4, 2.4**
        """
        source = _read_function_source(DRAFT_WORKER_PY,
                                       "_run_glue_decode")
        assert "[DISAGG_DIAG][CP3]" in source, (
            "CP3 MISSING: _run_glue_decode has no "
            "[DISAGG_DIAG][CP3] diagnostic logging for fused_ids, "
            "positions, per-token hs norms before fc projection."
        )

    def test_cp4_post_fc_logging(self):
        """Checkpoint 4: _run_glue_decode SHALL contain
        [DISAGG_DIAG][CP4] logging for post-fc projection norms.

        **Validates: Requirements 1.4, 2.4**
        """
        source = _read_function_source(DRAFT_WORKER_PY,
                                       "_run_glue_decode")
        assert "[DISAGG_DIAG][CP4]" in source, (
            "CP4 MISSING: _run_glue_decode has no "
            "[DISAGG_DIAG][CP4] diagnostic logging for per-token "
            "norms after fc projection."
        )

    def test_cp5_glue_decode_outputs_logging(self):
        """Checkpoint 5: _run_glue_decode SHALL contain
        [DISAGG_DIAG][CP5] logging for glue decode outputs.

        **Validates: Requirements 1.4, 2.4**
        """
        source = _read_function_source(DRAFT_WORKER_PY,
                                       "_run_glue_decode")
        assert "[DISAGG_DIAG][CP5]" in source, (
            "CP5 MISSING: _run_glue_decode has no "
            "[DISAGG_DIAG][CP5] diagnostic logging for prenorm "
            "norms per request, logits top-5 at recovery position."
        )

    def test_cp7_per_step_jit_logging(self):
        """Checkpoint 7: eagle_sequential_speculate SHALL contain
        [DISAGG_DIAG][CP7] logging for per-step JIT norms/logits.

        **Validates: Requirements 1.5, 2.5**
        """
        source = _read_function_source(DRAFT_MODEL_RUNNER_PY,
                                       "eagle_sequential_speculate")
        assert "[DISAGG_DIAG][CP7]" in source, (
            "CP7 MISSING: eagle_sequential_speculate has no "
            "[DISAGG_DIAG][CP7] diagnostic logging for per-step "
            "input hs norm, output prenorm norm, top-5 logits, "
            "sampled token, position."
        )

    def test_cp8_colocated_comparison_logging(self):
        """Checkpoint 8: EagleSpeculator.propose SHALL contain
        [COLOCATED_DIAG][CP8] logging for comparison with disagg.

        **Validates: Requirements 2.2**
        """
        source = _read_function_source(EAGLE_SPECULATOR_PY, "propose")
        assert "[COLOCATED_DIAG][CP8]" in source, (
            "CP8 MISSING: EagleSpeculator.propose has no "
            "[COLOCATED_DIAG][CP8] diagnostic logging for "
            "co-located comparison (raw hs norms, fc-projected "
            "norms, per-step hs norms, top-5 logits, sampled tokens)."
        )

    def test_cp9_fc_projection_logging(self):
        """Checkpoint 9: combine_hidden_states SHALL contain
        [DISAGG_DIAG][CP9] logging for fc projection internals.

        **Validates: Requirements 2.2**
        """
        source = _read_function_source(LLAMA_EAGLE3_PY,
                                       "combine_hidden_states")
        assert "[DISAGG_DIAG][CP9]" in source, (
            "CP9 MISSING: combine_hidden_states has no "
            "[DISAGG_DIAG][CP9] diagnostic logging for input norm, "
            "output norm, norm_before_fc flag, shape, dtype."
        )
