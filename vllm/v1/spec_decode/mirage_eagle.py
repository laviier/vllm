# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Mirage-accelerated EAGLE proposer for speculative decoding.

Uses Mirage's KNGraph API to compile fused CUDA kernels for the
non-attention portions of the EAGLE draft model forward pass:

  Pre-attention:  layernorm → QKV projection  (fused into 1 kernel)
  Post-attention: layernorm → gate_up → SiLU*Mul → down_proj  (fused into 1 kernel)

Attention runs through vLLM's standard backend (FlashAttention/FlashInfer)
with full KV cache support. This hybrid approach gets the fusion benefits
while keeping vLLM's battle-tested attention + KV cache infrastructure.

Requirements:
  - mirage package (pip install -e . from mirage repo, branch mpk)
  - NVIDIA GPU with SM >= 80

Usage:
  --speculative-config '{"method": "eagle3", ..., "use_mirage_draft": true}'
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.library import custom_op

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diagnostic counters — set MIRAGE_EAGLE_DEBUG=1 to enable verbose logging.
# ---------------------------------------------------------------------------
_DEBUG = os.environ.get("MIRAGE_EAGLE_DEBUG", "0") == "1"

_stats_lock = threading.Lock()
_stats: dict[str, int] = {
    "mirage_calls": 0,       # batch_size in compiled set, Mirage kernel used
    "fallback_calls": 0,     # batch_size not compiled, plain matmul used
    "mirage_empty": 0,       # Mirage returned empty outputs (fell through)
    "total_draft_tokens": 0, # tokens proposed by EAGLE
    "total_accepted": 0,     # tokens accepted by target model
}

# Number of inference calls to verify against reference matmul.
# Set MIRAGE_EAGLE_VERIFY_CALLS=N to verify the first N calls per layer.
# This is expensive (doubles the matmul work) but catches runtime correctness
# issues that don't show up in the startup correctness check.
_VERIFY_CALLS = int(os.environ.get("MIRAGE_EAGLE_VERIFY_CALLS", "0"))
_verify_call_counts: dict[int, int] = {}  # layer_idx -> call count

# MIRAGE_EAGLE_FALLBACK_ONLY=1 — always use original MLP, never Mirage kernel.
# FusedPostAttn wrappers are still installed but always call self._orig_mlp(x).
# Use this to test whether the Mirage kernel itself is causing acceptance issues.
_FALLBACK_ONLY = os.environ.get("MIRAGE_EAGLE_FALLBACK_ONLY", "0") == "1"

# MIRAGE_EAGLE_NO_WRAP=1 — skip installing FusedPostAttn wrappers entirely.
# Mirage kernels are compiled but never used. The model runs exactly as baseline.
# Use this to test whether the wrapper installation (not the kernel) causes issues.
_NO_WRAP = os.environ.get("MIRAGE_EAGLE_NO_WRAP", "0") == "1"

# MIRAGE_EAGLE_NO_COMPILE=1 — skip Mirage kernel compilation entirely.
# maybe_enable_mirage_eagle() returns immediately after this check.
# Use this to test whether Mirage's compile() calls corrupt GPU state.
_NO_COMPILE = os.environ.get("MIRAGE_EAGLE_NO_COMPILE", "0") == "1"


def _inc(key: str, n: int = 1) -> None:
    with _stats_lock:
        _stats[key] += n


def log_mirage_stats() -> None:
    """Log a summary of Mirage kernel usage and draft acceptance rate."""
    with _stats_lock:
        s = dict(_stats)
    total = s["mirage_calls"] + s["fallback_calls"]
    mirage_pct = 100.0 * s["mirage_calls"] / total if total else 0.0
    accept_rate = (
        100.0 * s["total_accepted"] / s["total_draft_tokens"]
        if s["total_draft_tokens"] else 0.0
    )
    logger.info(
        "[MirageEagle] kernel=mirage:%d fallback:%d (%.1f%% mirage) | "
        "empty_outputs:%d | draft_tokens:%d accepted:%d (%.1f%% accept_rate)",
        s["mirage_calls"], s["fallback_calls"], mirage_pct,
        s["mirage_empty"],
        s["total_draft_tokens"], s["total_accepted"], accept_rate,
    )


def _try_import_mirage():
    try:
        import mirage as mi
        return mi
    except ImportError:
        raise ImportError(
            "Mirage is required for MirageEagleProposer. "
            "Install: git clone --recursive --branch mpk "
            "https://github.com/mirage-project/mirage && "
            "cd mirage && pip install -e . -v"
        )


# --- Custom ops for Mirage fused EAGLE MLP ---
# Both ops are registered as torch custom ops so that Dynamo / aot_compile
# can trace through them without graph breaks or ConstraintViolationErrors.
# The actual dispatch logic lives inside the op implementations.

# Global storage for compiled Mirage graphs.
# Keyed by (registry_id, batch_size) → (graph, out_dim).
# All EAGLE layers share the same weight *shape*, so one compiled graph
# per batch size is sufficient. The actual weight tensor is passed at
# cuda_call time (looked up from _per_layer_weights by layer_idx).
# This keeps compilation time at O(num_batch_sizes) instead of
# O(num_layers × num_batch_sizes).
_mirage_gate_up_registry: dict[tuple[int, int], tuple] = {}  # (id, bs) -> (graph, out_dim)

# Global storage for original MLP callables, used by mirage::mlp_forward
# fallback path.  Keyed by (registry_id, layer_idx) → original nn.Module.
# Populated in maybe_enable_mirage_eagle() before any forward pass.
_orig_mlp_registry: dict[tuple[int, int], nn.Module] = {}

# Global storage for (act_fn, down_proj) pairs used by the Mirage fast path
# inside mirage::mlp_forward.  Keyed by (registry_id, layer_idx).
_mlp_parts_registry: dict[tuple[int, int], tuple] = {}  # → (act_fn, down_proj)

# Batch sizes to pre-compile Mirage kernels for.
# EAGLE CUDA graph capture uses multiples of uniform_decode_query_len
# (= num_speculative_tokens + 1). For num_speculative_tokens=5, that's
# multiples of 6. We compile a dense set of small sizes (where most
# real traffic lands) and sparse coverage at larger sizes.
# Sizes not in this list fall back to plain matmul automatically.
# Override via MIRAGE_EAGLE_COMPILE_BATCH_SIZES env var (comma-separated).
_DEFAULT_COMPILE_BATCH_SIZES = (
    [1]
    + list(range(6, 132, 6))    # 6,12,...,126  (dense: 21 sizes)
    + list(range(132, 513, 12)) # 132,144,...,504 (sparse: 32 sizes)
)

def _get_compile_batch_sizes() -> list[int]:
    env = os.environ.get("MIRAGE_EAGLE_COMPILE_BATCH_SIZES", "")
    if env:
        try:
            return sorted(set(int(x.strip()) for x in env.split(",") if x.strip()))
        except ValueError:
            logger.warning(
                "Invalid MIRAGE_EAGLE_COMPILE_BATCH_SIZES=%r, using defaults", env
            )
    return _DEFAULT_COMPILE_BATCH_SIZES

_MIRAGE_COMPILE_BATCH_SIZES = _get_compile_batch_sizes()


@custom_op("mirage::fused_gate_up", mutates_args=())
def _fused_gate_up(x: torch.Tensor, registry_id: int, layer_idx: int) -> torch.Tensor:
    """Fused gate_up_proj matmul via Mirage compiled kernel.

    The compiled graph is looked up by (registry_id, batch_size).
    The weight tensor is looked up by layer_idx from _per_layer_weights
    on the MirageEagleFusedKernels instance — so each layer uses its own
    weights while sharing the compiled kernel structure.
    """
    batch_size = x.shape[0]
    entry = _mirage_gate_up_registry.get((registry_id, batch_size))
    if entry is not None:
        fk = MirageEagleFusedKernels._registry.get(registry_id)
        if fk is None:
            raise RuntimeError(
                f"Mirage registry_id {registry_id} not found in class registry"
            )
        w_gateup_t = fk._per_layer_weights[layer_idx]
        graph, out_dim = entry
        if _FALLBACK_ONLY:
            # Diagnostic mode: FusedPostAttn should have bypassed this call.
            # Return zero-numel sentinel to trigger fallback in caller.
            _inc("fallback_calls")
            if _DEBUG:
                logger.info(
                    "[MirageEagle] FALLBACK_ONLY mode: returning sentinel "
                    "layer=%d bs=%d", layer_idx, batch_size,
                )
            return x.new_empty(0)
        _inc("mirage_calls")
        if _DEBUG:
            logger.info(
                "[MirageEagle] fused_gate_up: Mirage kernel path, "
                "layer=%d shape=%s capturing=%s",
                layer_idx, list(x.shape),
                torch.cuda.is_current_stream_capturing(),
            )
        outputs = graph.cuda_call(inputs=[x, w_gateup_t])
        if outputs:
            result = outputs[0]
            # Ensure contiguous row-major layout — Mirage may return column-major.
            if not result.is_contiguous():
                result = result.contiguous()
            if _DEBUG:
                logger.info(
                    "[MirageEagle] fused_gate_up: Mirage kernel returned shape=%s",
                    list(result.shape),
                )
            # Runtime verification: compare against reference matmul for the
            # first MIRAGE_EAGLE_VERIFY_CALLS calls per layer.
            # Note: this runs during CUDA graph capture (warmup), not replay.
            if _VERIFY_CALLS > 0 and not torch.cuda.is_current_stream_capturing():
                count = _verify_call_counts.get(layer_idx, 0)
                if count < _VERIFY_CALLS:
                    _verify_call_counts[layer_idx] = count + 1
                    ref = x @ w_gateup_t
                    max_diff = (result - ref).abs().max().item()
                    mean_diff = (result - ref).abs().mean().item()
                    logger.info(
                        "[MirageEagle] RUNTIME VERIFY layer=%d bs=%d call=%d: "
                        "max_diff=%.6f mean_diff=%.6f x_norm=%.4f",
                        layer_idx, batch_size, count,
                        max_diff, mean_diff, x.float().norm().item(),
                    )
                    if max_diff > 0.5:
                        logger.error(
                            "[MirageEagle] RUNTIME CORRECTNESS FAILURE layer=%d "
                            "bs=%d: max_diff=%.4f — Mirage output is WRONG during "
                            "actual inference!", layer_idx, batch_size, max_diff,
                        )
            return result
        # cuda_call returned empty — return zero-numel sentinel so
        # FusedPostAttn falls back to the original MLP (which handles
        # quantization, bias, and TP correctly).
        _inc("mirage_empty")
        logger.warning(
            "[MirageEagle] cuda_call returned empty outputs for layer=%d "
            "batch_size=%d. Falling back to original MLP.",
            layer_idx, batch_size,
        )
        return x.new_empty(0)
    else:
        _inc("fallback_calls")
        if _DEBUG:
            logger.info(
                "[MirageEagle] fused_gate_up: no compiled kernel for "
                "batch_size=%d — FusedPostAttn should have caught this",
                batch_size,
            )
    # Should not reach here: FusedPostAttn checks has_kernel before calling.
    # Return zero-numel sentinel to trigger fallback in caller.
    return x.new_empty(0)


@_fused_gate_up.register_fake
def _fused_gate_up_fake(x: torch.Tensor, registry_id: int, layer_idx: int) -> torch.Tensor:
    # Look up out_dim from any compiled entry for this registry_id
    for (rid, _bs), (_, out_dim) in _mirage_gate_up_registry.items():
        if rid == registry_id:
            return x.new_empty(x.shape[0], out_dim)
    # Fallback: return zero-numel sentinel (triggers MLP fallback in caller)
    return x.new_empty(0)


# --- mirage::mlp_forward custom op ---
# Encapsulates the full MLP dispatch: Mirage fast path (gate_up via compiled
# kernel + act_fn + down_proj) or fallback to the original MLP module.
# Registered as a custom op so aot_compile / Dynamo treats it as an opaque
# leaf — no graph breaks, no ConstraintViolationError from shape specialization.
#
# The original MLP and (act_fn, down_proj) parts are stored in module-level
# registries keyed by (registry_id, layer_idx), populated before any forward.

@custom_op("mirage::mlp_forward", mutates_args=())
def _mlp_forward(x: torch.Tensor, registry_id: int, layer_idx: int) -> torch.Tensor:
    """Full MLP dispatch: Mirage gate_up (if compiled for this batch size)
    followed by act_fn + down_proj, or fallback to the original MLP.

    This is the single entry point called from FusedPostAttn.forward().
    Dynamo sees it as an opaque custom op — no shape specialization issues.

    NOTE: We do NOT call torch.ops.mirage.fused_gate_up here. Custom ops
    must not call other custom ops in their implementation — doing so causes
    silent mis-dispatch under aot_compile. Instead we inline the Mirage
    cuda_call directly.

    NOTE: We skip Mirage during CUDA graph capture/replay. Mirage's cuda_call
    does not support CUDA graph capture — it launches kernels directly and
    cannot be recorded into a CUDA graph. Calling it during capture produces
    a graph that replays stale outputs. We fall back to the original MLP for
    all captured batch sizes; Mirage only runs in eager (non-captured) mode.
    """
    # Skip Mirage entirely during CUDA graph capture or replay.
    # torch.cuda.is_current_stream_capturing() returns True during capture.
    if torch.cuda.is_current_stream_capturing():
        orig_mlp = _orig_mlp_registry.get((registry_id, layer_idx))
        if orig_mlp is not None:
            return orig_mlp(x)
        raise RuntimeError(
            f"[MirageEagle] No MLP found for registry_id={registry_id} "
            f"layer_idx={layer_idx} during CUDA graph capture."
        )

    batch_size = x.shape[0]
    entry = _mirage_gate_up_registry.get((registry_id, batch_size))
    if entry is not None and not _FALLBACK_ONLY:
        fk = MirageEagleFusedKernels._registry.get(registry_id)
        if fk is not None:
            w_gateup_t = fk._per_layer_weights[layer_idx]
            graph, _out_dim = entry
            _inc("mirage_calls")
            if _DEBUG:
                logger.info(
                    "[MirageEagle] mlp_forward: Mirage path layer=%d bs=%d",
                    layer_idx, batch_size,
                )
            outputs = graph.cuda_call(inputs=[x, w_gateup_t])
            if outputs:
                gateup = outputs[0]
                if not gateup.is_contiguous():
                    gateup = gateup.contiguous()
                parts = _mlp_parts_registry.get((registry_id, layer_idx))
                if parts is not None:
                    act_fn, down_proj = parts
                    out = act_fn(gateup)
                    out, _ = down_proj(out)
                    return out
            # cuda_call returned empty — fall through to original MLP
            _inc("mirage_empty")
            logger.warning(
                "[MirageEagle] cuda_call returned empty for layer=%d bs=%d, "
                "falling back to original MLP.", layer_idx, batch_size,
            )
    else:
        _inc("fallback_calls")
        if _DEBUG:
            logger.info(
                "[MirageEagle] mlp_forward: fallback path layer=%d bs=%d "
                "(FALLBACK_ONLY=%s, has_entry=%s)",
                layer_idx, batch_size, _FALLBACK_ONLY, entry is not None,
            )
        _inc("fallback_calls")
        if _DEBUG:
            logger.info(
                "[MirageEagle] mlp_forward: fallback path layer=%d bs=%d "
                "(FALLBACK_ONLY=%s, has_entry=%s)",
                layer_idx, batch_size, _FALLBACK_ONLY, entry is not None,
            )
    # Fallback: original MLP (handles quantization, bias, TP correctly)
    orig_mlp = _orig_mlp_registry.get((registry_id, layer_idx))
    if orig_mlp is not None:
        return orig_mlp(x)
    raise RuntimeError(
        f"[MirageEagle] No MLP found for registry_id={registry_id} "
        f"layer_idx={layer_idx}. Was maybe_enable_mirage_eagle() called?"
    )


@_mlp_forward.register_fake
def _mlp_forward_fake(x: torch.Tensor, registry_id: int, layer_idx: int) -> torch.Tensor:
    # Output shape is [num_tokens, hidden_size] — same as input.
    # (down_proj maps intermediate_size → hidden_size, and x enters as hidden_size)
    return x.new_empty(x.shape[0], x.shape[1])


class MirageEagleFusedKernels:
    """Compiles fused Mirage kernels for EAGLE head non-attention ops.

    Two kernels are compiled:
      pre_attn:  layernorm → QKV projection
      post_attn: layernorm → gate_up_proj → SiLU → down_proj

    Each replaces multiple separate CUDA kernel launches with a single
    fused launch. The embed + fc + attention + O-proj + residual + final
    norm + lm_head still run via PyTorch (attention needs vLLM's KV cache).
    """

    # Class-level registry for compiled graphs so the custom op can look
    # them up by integer key (custom ops can't capture arbitrary objects).
    _registry: dict[int, "MirageEagleFusedKernels"] = {}
    _next_id: int = 0

    def __init__(
        self,
        model: nn.Module,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        mi = _try_import_mirage()
        self.mi = mi
        self.device = device

        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        self.draft_config = spec_config.draft_model_config.hf_config
        self.hidden_size = spec_config.draft_model_config.get_hidden_size()

        inner = model.model if hasattr(model, "model") else model
        self.inner_model = inner

        num_heads = getattr(self.draft_config, "num_attention_heads", 32)
        num_kv_heads = getattr(self.draft_config, "num_key_value_heads", 8)
        head_dim = getattr(
            self.draft_config, "head_dim", self.hidden_size // num_heads
        )
        self.intermediate_size = getattr(
            self.draft_config, "intermediate_size", 4 * self.hidden_size
        )
        self.qkv_size = (num_heads + 2 * num_kv_heads) * head_dim

        self.pre_attn_compiled = None   # compiled KNGraph result dict
        self.pre_attn_graph = None      # KNGraph with .run set
        self.post_attn_compiled = None
        self.post_attn_graph = None
        self._compiled = False

        # Per-layer weight tensors: _per_layer_weights[layer_idx] = w_gateup_t
        # Populated during compile() without triggering extra compilations.
        self._per_layer_weights: list[torch.Tensor] = []

        # Register this instance so the custom op can find it
        self._registry_id = MirageEagleFusedKernels._next_id
        MirageEagleFusedKernels._next_id += 1
        MirageEagleFusedKernels._registry[self._registry_id] = self

    def compile(self) -> None:
        """Build and compile fused kernels for all EAGLE decoder layers."""
        mi = self.mi
        inner = self.inner_model
        layers = list(inner.layers)

        # Read weight shapes from layer 0 (all layers share the same shape)
        layer0 = layers[0]
        qkv_weight = layer0.self_attn.qkv_proj.weight.data
        qkv_out_dim, qkv_in_dim = qkv_weight.shape

        gateup_weight = layer0.mlp.gate_up_proj.weight.data
        gateup_out_dim, gateup_in_dim = gateup_weight.shape

        down_weight = layer0.mlp.down_proj.weight.data
        down_out_dim, down_in_dim = down_weight.shape

        logger.info(
            "EAGLE layer weights: qkv=%s, gate_up=%s, down=%s",
            list(qkv_weight.shape), list(gateup_weight.shape),
            list(down_weight.shape),
        )

        # ---- Pre-attention: layernorm → QKV projection ----
        # NOTE: Mirage's KNGraph rms_norm has a Z3 solver bug that causes
        # crashes for many hidden sizes. Skip pre-attention fusion for now.
        has_layernorm = not isinstance(
            getattr(layer0, "input_layernorm", None), nn.Identity
        )
        if has_layernorm:
            logger.info(
                "Skipping pre-attention fusion (rms_norm not supported in "
                "KNGraph for hidden_size=%d)", qkv_in_dim,
            )

        # ---- Collect per-layer weights (no compilation needed) ----
        # All layers share the same weight shape, so we only compile the
        # KNGraph structure once per batch size (using layer 0's weight for
        # the compile call). Each layer's actual w_gateup_t is stored in
        # _per_layer_weights and passed at cuda_call time.
        for layer in layers:
            w_gateup_t = layer.mlp.gate_up_proj.weight.data.t().contiguous()
            self._per_layer_weights.append(w_gateup_t)

        logger.info(
            "Collected gate_up weights for %d EAGLE layers", len(layers)
        )

        # ---- Post-attention: gate_up matmul, compiled once per batch size ----
        # Use layer 0's weight for the compile call (shape is all that matters).
        w_gateup_t_l0 = self._per_layer_weights[0]

        try:
            compiled_any_bs = False
            for bs in _MIRAGE_COMPILE_BATCH_SIZES:
                try:
                    g = mi.new_kernel_graph()
                    h_in = g.new_input(
                        dims=(bs, gateup_in_dim), dtype=mi.bfloat16
                    )
                    w_in = g.new_input(
                        dims=(gateup_in_dim, gateup_out_dim), dtype=mi.bfloat16
                    )
                    out = g.matmul(h_in, w_in)
                    g.mark_output(out)

                    h_dummy = torch.zeros(
                        bs, gateup_in_dim,
                        dtype=torch.bfloat16, device=self.device,
                    )
                    result = g.compile(inputs=[h_dummy, w_gateup_t_l0])
                    if result is not None and g.run is not None:
                        _mirage_gate_up_registry[
                            (self._registry_id, bs)
                        ] = (g, gateup_out_dim)
                        compiled_any_bs = True
                        logger.info(
                            "Mirage MLP gate_up kernel compiled for batch_size=%d", bs
                        )
                    else:
                        logger.warning(
                            "Mirage MLP compile returned None for batch_size=%d", bs
                        )
                except Exception as e:
                    logger.warning(
                        "Failed to compile MLP kernel for batch_size=%d: %s", bs, e
                    )

            if compiled_any_bs:
                self.post_attn_graph = True  # sentinel: at least one size compiled
                self.down_out_dim = down_out_dim
                self.down_in_dim = down_in_dim

                # Eager correctness check: test each layer at multiple batch sizes
                # with fixed deterministic input against actual model weights.
                # Any batch size that fails is removed from the registry.
                _CHECK_BATCH_SIZES = [bs for bs in [1, 6, 12, 48]
                                      if (self._registry_id, bs) in _mirage_gate_up_registry]
                failed_bs: list[int] = []

                for layer_idx, w_t in enumerate(self._per_layer_weights):
                    for test_bs in _CHECK_BATCH_SIZES:
                        if test_bs in failed_bs:
                            continue  # already marked failed
                        try:
                            entry = _mirage_gate_up_registry.get(
                                (self._registry_id, test_bs)
                            )
                            if entry is None:
                                continue
                            g_test, _ = entry
                            torch.manual_seed(42)
                            test_in = torch.randn(
                                test_bs, gateup_in_dim,
                                dtype=torch.bfloat16, device=self.device,
                            )
                            mirage_out = g_test.cuda_call(inputs=[test_in, w_t])
                            if not mirage_out:
                                logger.warning(
                                    "[MirageEagle] Correctness check layer=%d bs=%d: "
                                    "cuda_call returned empty", layer_idx, test_bs,
                                )
                                failed_bs.append(test_bs)
                                continue
                            mirage_result = mirage_out[0]
                            if not mirage_result.is_contiguous():
                                mirage_result = mirage_result.contiguous()
                            ref_out = test_in @ w_t
                            max_diff = (mirage_result - ref_out).abs().max().item()
                            mean_diff = (mirage_result - ref_out).abs().mean().item()
                            logger.info(
                                "[MirageEagle] Correctness check layer=%d bs=%d: "
                                "strides=%s max_diff=%.6f mean_diff=%.6f",
                                layer_idx, test_bs,
                                mirage_result.stride(), max_diff, mean_diff,
                            )
                            if max_diff > 0.5 or (max_diff != max_diff):  # NaN check
                                logger.warning(
                                    "[MirageEagle] CORRECTNESS FAILURE layer=%d bs=%d: "
                                    "max_diff=%.4f — disabling this batch size.",
                                    layer_idx, test_bs, max_diff,
                                )
                                failed_bs.append(test_bs)
                        except Exception as e:
                            logger.warning(
                                "[MirageEagle] Correctness check layer=%d bs=%d failed: %s",
                                layer_idx, test_bs, e,
                            )
                            failed_bs.append(test_bs)

                # Remove failing batch sizes from the registry
                for bs in failed_bs:
                    _mirage_gate_up_registry.pop((self._registry_id, bs), None)
                    logger.info(
                        "[MirageEagle] Removed bs=%d from registry "
                        "(correctness failure)", bs
                    )

                # If all checked sizes failed, disable post_attn entirely
                remaining = [bs for (rid, bs) in _mirage_gate_up_registry
                             if rid == self._registry_id]
                if not remaining:
                    logger.warning(
                        "[MirageEagle] ALL batch sizes failed correctness check. "
                        "Disabling Mirage gate_up fusion entirely. "
                        "Rebuild Mirage: cd ~/github/mirage && pip install -e . -v"
                    )
                    self.post_attn_graph = None
            else:
                logger.warning("Mirage MLP compile failed for all batch sizes")
        except Exception as e:
            logger.warning("Failed to compile MLP kernel: %s", e)

        self._compiled = (
            self.pre_attn_graph is not None
            or bool(self.post_attn_graph)
        )
        if self._compiled:
            compiled_bs = sorted(
                bs for (rid, bs) in _mirage_gate_up_registry
                if rid == self._registry_id
            )
            logger.info(
                "Mirage fused kernels ready: pre_attn=%s, post_attn=%s "
                "(layers=%d, batch_sizes=%s)",
                self.pre_attn_graph is not None,
                bool(self.post_attn_graph),
                len(layers),
                compiled_bs,
            )

    def run_pre_attn(
        self,
        hidden_states: torch.Tensor,
        w_qkv: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run fused layernorm + QKV projection.

        Args:
            hidden_states: [num_tokens, hidden_size]
            w_qkv: QKV weight tensor [qkv_size, hidden_size]

        Returns:
            qkv: [num_tokens, qkv_size] or None if not compiled.
        """
        if self.pre_attn_graph is None:
            return None
        outputs = self.pre_attn_graph.cuda_call(
            inputs=[hidden_states, w_qkv]
        )
        return outputs[0] if outputs else None

    def run_post_attn(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor | None:
        """Run fused gate_up_proj matmul via Mirage custom op.

        Args:
            hidden_states: [num_tokens, hidden_size]
            layer_idx: which EAGLE decoder layer this is for

        Returns:
            gate_up output: [num_tokens, 2*intermediate] or None if Mirage
            failed (caller should fall back to original MLP).
        """
        if not self.post_attn_graph:
            return None
        result = torch.ops.mirage.fused_gate_up(
            hidden_states, self._registry_id, layer_idx
        )
        # _fused_gate_up returns a zero-numel tensor as a sentinel when
        # cuda_call returns empty (Mirage kernel not ready).
        if result.numel() == 0:
            return None
        return result


class MirageEagleProposer:
    """Wrapper that holds compiled Mirage fused kernels for EAGLE.

    This doesn't replace propose() — it provides fused kernel alternatives
    that the EAGLE proposer's model forward pass can call instead of
    individual PyTorch ops.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device
        self.fused_kernels: MirageEagleFusedKernels | None = None
        self._compiled = False

    def load_model(self, eagle_model: nn.Module) -> None:
        self.fused_kernels = MirageEagleFusedKernels(
            model=eagle_model,
            vllm_config=self.vllm_config,
            device=self.device,
        )
        self.fused_kernels.compile()
        self._compiled = self.fused_kernels._compiled

    @staticmethod
    def is_available() -> bool:
        try:
            import mirage  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def wrap(base_proposer, vllm_config, device) -> "MirageEagleProposer":
        proposer = MirageEagleProposer(vllm_config, device)
        if hasattr(base_proposer, "model") and base_proposer.model is not None:
            proposer.load_model(base_proposer.model)
        return proposer


def _run_live_correctness_check(fk: "MirageEagleFusedKernels") -> None:
    """Run a live correctness check with random hidden states.

    Called once after kernel installation, outside of any compiled graph,
    so torch.cuda.is_current_stream_capturing() is safe to call here.

    Also verifies that cuda_call actually uses the weight passed at call time
    (not the weight baked in at compile time). If Mirage ignores the runtime
    weight, all layers would produce identical outputs — which would explain
    a near-zero acceptance rate.
    """
    if torch.cuda.is_current_stream_capturing():
        return

    # Basic per-layer correctness check
    for layer_idx, w_t in enumerate(fk._per_layer_weights):
        in_dim = w_t.shape[0]
        for test_bs in [1, 6]:
            entry = _mirage_gate_up_registry.get((fk._registry_id, test_bs))
            if entry is None:
                continue
            g, _ = entry
            torch.manual_seed(0)
            x = torch.randn(test_bs, in_dim, dtype=torch.bfloat16, device=fk.device)
            outputs = g.cuda_call(inputs=[x, w_t])
            if not outputs:
                logger.warning(
                    "[MirageEagle] LIVE check layer=%d bs=%d: cuda_call empty",
                    layer_idx, test_bs,
                )
                continue
            result = outputs[0]
            if not result.is_contiguous():
                result = result.contiguous()
            ref = x @ w_t
            max_diff = (result - ref).abs().max().item()
            logger.info(
                "[MirageEagle] LIVE check layer=%d bs=%d: "
                "x.dtype=%s x.shape=%s max_diff=%.6f",
                layer_idx, test_bs, x.dtype, list(x.shape), max_diff,
            )

    # Cross-layer weight isolation check: verify that passing layer N's weight
    # actually produces layer N's output (not layer 0's output baked at compile).
    # If Mirage ignores the runtime weight argument, result_l0 == result_lN
    # even though w_l0 != w_lN — this would cause all layers to use layer 0's
    # weights and produce wrong hidden states, killing acceptance rate.
    if len(fk._per_layer_weights) >= 2:
        entry = _mirage_gate_up_registry.get((fk._registry_id, 6))
        if entry is not None:
            g, _ = entry
            w_l0 = fk._per_layer_weights[0]
            w_l1 = fk._per_layer_weights[1]
            in_dim = w_l0.shape[0]
            torch.manual_seed(1)
            x = torch.randn(6, in_dim, dtype=torch.bfloat16, device=fk.device)

            out_l0 = g.cuda_call(inputs=[x, w_l0])
            out_l1 = g.cuda_call(inputs=[x, w_l1])

            if out_l0 and out_l1:
                r0 = out_l0[0]
                r1 = out_l1[0]
                if not r0.is_contiguous(): r0 = r0.contiguous()
                if not r1.is_contiguous(): r1 = r1.contiguous()
                diff_between_layers = (r0 - r1).abs().max().item()
                ref_l0 = x @ w_l0
                ref_l1 = x @ w_l1
                ref_diff = (ref_l0 - ref_l1).abs().max().item()
                logger.info(
                    "[MirageEagle] WEIGHT ISOLATION CHECK: "
                    "diff(mirage_l0, mirage_l1)=%.6f  ref_diff=%.6f  "
                    "weight_isolation=%s",
                    diff_between_layers, ref_diff,
                    "OK" if diff_between_layers > 1e-4 else "FAIL — Mirage ignores runtime weight!",
                )
                if diff_between_layers < 1e-4 and ref_diff > 1e-4:
                    logger.error(
                        "[MirageEagle] CRITICAL: Mirage cuda_call ignores the weight "
                        "passed at runtime and always uses the weight from compile time "
                        "(layer 0). All EAGLE layers are using layer 0's weights. "
                        "This explains the ~0%% acceptance rate. "
                        "The compiled graph must be re-compiled per layer, or Mirage "
                        "must support runtime weight binding."
                    )


def maybe_enable_mirage_eagle(
    proposer,
    vllm_config: VllmConfig,
    device: torch.device,
) -> None:
    """Optionally enable Mirage fused kernels for the EAGLE draft model."""
    spec_config = vllm_config.speculative_config
    if spec_config is None:
        return

    use_mirage = getattr(spec_config, "use_mirage_draft", False)
    if not use_mirage:
        return

    if not MirageEagleProposer.is_available():
        logger.warning(
            "use_mirage_draft=True but mirage is not installed. "
            "Falling back to standard execution."
        )
        return

    if not spec_config.use_eagle():
        logger.warning(
            "Mirage draft acceleration only supports EAGLE/MTP. "
            "Method: %s. Falling back.", spec_config.method,
        )
        return

    if getattr(spec_config, "parallel_drafting", False):
        logger.info(
            "use_mirage_draft=True but parallel_drafting=True: skipping Mirage "
            "integration. Disable parallel_drafting to use Mirage acceleration."
        )
        return

    if _NO_COMPILE:
        logger.info(
            "[MirageEagle] MIRAGE_EAGLE_NO_COMPILE=1: skipping Mirage compilation "
            "entirely. Model runs as baseline. Use this to test whether "
            "Mirage's compile() calls corrupt GPU state."
        )
        return

    try:
        mirage_proposer = MirageEagleProposer.wrap(
            proposer, vllm_config, device
        )
        proposer._mirage_proposer = mirage_proposer

        # Install fused kernels into the EAGLE model's decoder layers.
        fk = mirage_proposer.fused_kernels
        if fk is not None and fk._compiled:
            if _NO_WRAP:
                logger.info(
                    "[MirageEagle] MIRAGE_EAGLE_NO_WRAP=1: Mirage kernels compiled "
                    "but FusedPostAttn wrappers NOT installed. Model runs as baseline. "
                    "Use this to test whether wrapper installation causes issues."
                )
            else:
                inner = (
                    proposer.model.model
                    if hasattr(proposer.model, "model")
                    else proposer.model
                )
                for i, layer in enumerate(inner.layers):
                    if fk.post_attn_graph is not None and fk.post_attn_graph:
                        original_mlp = layer.mlp
                        layer_idx = i  # capture per-iteration

                        # Register original MLP and its parts for the custom op
                        _orig_mlp_registry[(fk._registry_id, layer_idx)] = original_mlp
                        _mlp_parts_registry[(fk._registry_id, layer_idx)] = (
                            original_mlp.act_fn,
                            original_mlp.down_proj,
                        )

                        class FusedPostAttn(nn.Module):
                            """Drop-in replacement for LlamaMLP.

                            Delegates entirely to torch.ops.mirage.mlp_forward,
                            which is a registered custom op. Dynamo treats it as
                            an opaque leaf — no graph breaks, no shape
                            specialization conflicts with aot_compile.
                            """

                            def __init__(self, registry_id, layer_idx):
                                super().__init__()
                                self._registry_id = registry_id
                                self._layer_idx = layer_idx
                                self._call_count = 0

                            def forward(self, x):
                                self._call_count += 1
                                if self._call_count % 500 == 0 and _DEBUG:
                                    log_mirage_stats()
                                return torch.ops.mirage.mlp_forward(
                                    x, self._registry_id, self._layer_idx
                                )

                        layer.mlp = FusedPostAttn(fk._registry_id, layer_idx)
                        logger.info(
                            "Installed Mirage fused MLP on EAGLE layer %d", i,
                        )

                # One-time live correctness check with real hidden states,
                # done here (outside forward) so Dynamo never sees it.
                if _DEBUG and fk is not None and fk._compiled:
                    _run_live_correctness_check(fk)

        logger.info("Mirage EAGLE fused kernels enabled.")
        logger.info(
            "[MirageEagle] Diagnostics active. Set MIRAGE_EAGLE_DEBUG=1 for "
            "per-call verbose logging. Call log_mirage_stats() for a summary."
        )
        if _DEBUG:
            logger.info(
                "[MirageEagle] DEBUG mode on. Will log kernel path on every call "
                "and print stats every 500 FusedPostAttn forward calls."
            )
    except Exception:
        logger.warning(
            "Failed to compile Mirage EAGLE kernels. Falling back.",
            exc_info=True,
        )
