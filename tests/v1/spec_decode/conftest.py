# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local conftest for spec_decode tests.

Patches the broken torch._dynamo.convert_frame.GraphCaptureOutput import
that occurs with torch < 2.12 in the current vllm env_override.py.
This allows tests that only depend on lightweight modules (data models,
draft_server skeleton, draft_router) to run without a full vllm build.
"""
import sys

# Patch before any vllm import triggers env_override.py
try:
    from torch._dynamo.convert_frame import GraphCaptureOutput  # noqa: F401
except ImportError:
    import torch._dynamo.convert_frame as _cf

    class _StubGraphCaptureOutput:
        @staticmethod
        def get_runtime_env():
            return {}

    _cf.GraphCaptureOutput = _StubGraphCaptureOutput  # type: ignore[attr-defined]
