# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Saguaro Sampling for disagg draft (Speculative Speculative Decoding).

Saguaro sampling is a modified draft sampling scheme that deliberately
suppresses the probabilities of the top-F tokens (which are cached as
bonus token candidates). This biases the residual distribution used in
rejection sampling to concentrate on exactly those cached tokens, thereby
increasing the speculation cache hit rate at the cost of slightly lower
per-token acceptance rate.

The key insight: standard sampling produces bonus tokens from the
residual distribution max(p_target - p_draft, 0) / Z. If we suppress
p_draft on the cached tokens (multiply by C < 1), the residual
p_target - C*p_draft becomes larger for those tokens, making them more
likely to be the actual bonus token → higher cache hit rate.

Hyperparameter C ∈ [0, 1]:
  - C = 1.0: standard sampling (no modification)
  - C = 0.0: draft never samples cached tokens → maximum cache hit but
             lowest acceptance rate
  - C ≈ 0.3-0.5: good tradeoff (paper recommendation)

Reference: SSD paper §4.2 (Saguaro Sampling)
Reference impl: ssd/utils/async_helpers/async_spec_helpers.py::apply_sampler_x_rescaling
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def apply_saguaro_rescaling(
    logits: torch.Tensor,
    cached_token_indices: torch.Tensor | None = None,
    num_cached_tokens: int = 0,
    saguaro_c: float = 1.0,
) -> torch.Tensor:
    """Apply Saguaro probability rescaling to draft logits.

    Suppresses the probabilities of the top-F cached tokens by factor C,
    then renormalizes. This operates in probability space (softmax first),
    rescales, renormalizes, then converts back to logit space.

    This function is designed to be called on the draft model's logits
    before sampling, so that the draft avoids sampling the bonus token
    candidates we've already cached — pushing those tokens into the
    residual distribution where they increase cache hit rate.

    Args:
        logits: [B, V] — raw draft logits before sampling.
        cached_token_indices: [B, F] or None — indices of the F cached
            bonus token candidates per sequence. If None, uses top-F
            from the logits themselves.
        num_cached_tokens: F, the number of cached tokens to suppress.
            Only used if cached_token_indices is None.
        saguaro_c: C parameter ∈ [0, 1]. Controls suppression strength.
            1.0 = no change (standard sampling).
            0.0 = maximum suppression.

    Returns:
        Modified logits [B, V] with Saguaro rescaling applied.
    """
    if saguaro_c >= 1.0:
        # No modification needed
        return logits

    if cached_token_indices is None and num_cached_tokens <= 0:
        return logits

    B, V = logits.shape

    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1)

    if cached_token_indices is None:
        # Use top-F tokens from the logits as the cached set
        _, cached_token_indices = torch.topk(
            logits, num_cached_tokens + 1, dim=-1
        )

    # Create mask for cached positions
    F = cached_token_indices.shape[1]
    cache_mask = torch.zeros(B, V, dtype=torch.bool, device=logits.device)
    cache_mask.scatter_(dim=1, index=cached_token_indices, value=True)

    # Apply suppression: multiply cached token probs by C
    probs = torch.where(cache_mask, probs * saguaro_c, probs)

    # Renormalize
    prob_sum = probs.sum(dim=-1, keepdim=True)
    probs = probs / prob_sum.clamp(min=1e-10)

    # Convert back to logits
    # log(probs) with numerical stability
    modified_logits = torch.log(probs.clamp(min=1e-10))

    return modified_logits


def compute_saguaro_temperature(
    base_temperature: float,
    acceptance_rate: float,
    fan_out: int,
) -> float:
    """Compute effective Saguaro C parameter from model/config parameters.

    The paper suggests C should be tuned based on:
    - Temperature: higher T → more random → need stronger suppression
    - Acceptance rate: lower a → more rejections → more bonus token diversity
    - Fan-out: higher F → more cached tokens → can afford less suppression

    This is a heuristic based on the paper's experimental findings.

    Args:
        base_temperature: Sampling temperature T.
        acceptance_rate: Estimated acceptance rate a.
        fan_out: Number of cached bonus token candidates F.

    Returns:
        Recommended C parameter ∈ [0, 1].
    """
    if base_temperature <= 0:
        # Greedy decoding: C doesn't matter much since top-1 is deterministic
        # Use mild suppression to handle ties
        return 0.5

    # Heuristic: lower C when temperature is high (more randomness)
    # and when fan_out is small (fewer cached options)
    # C = 0.3 is paper's default for T=0 with F=3
    c = 0.3 + 0.2 * (1.0 / (1.0 + base_temperature)) + 0.1 * min(fan_out / 8.0, 1.0)
    return max(0.1, min(1.0, c))


class SaguaroSampler:
    """Saguaro-modified sampler for the draft model.

    Wraps the draft model's sampling to apply Saguaro rescaling
    before token selection, increasing speculation cache hit rate.

    Args:
        saguaro_c: C parameter for probability suppression (0-1).
            Use None for automatic selection based on temperature.
        fan_out: Number of cached bonus tokens per position.
        device: CUDA device.
    """

    def __init__(
        self,
        saguaro_c: float | None = None,
        fan_out: int = 3,
        device: torch.device = torch.device("cuda"),
    ):
        self.saguaro_c = saguaro_c
        self.fan_out = fan_out
        self.device = device

    def apply(
        self,
        logits: torch.Tensor,
        cached_token_indices: torch.Tensor | None = None,
        temperature: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply Saguaro rescaling to draft logits.

        Args:
            logits: [B, V] — raw draft logits.
            cached_token_indices: [B, F] — cached bonus token indices.
            temperature: Sampling temperature (for auto C selection).

        Returns:
            Modified logits with Saguaro rescaling applied.
        """
        # Determine C parameter
        if self.saguaro_c is not None:
            c = self.saguaro_c
        elif temperature is not None:
            if isinstance(temperature, torch.Tensor):
                # Use mean temperature for batch
                t = float(temperature.mean().item())
            else:
                t = float(temperature)
            c = compute_saguaro_temperature(t, 0.65, self.fan_out)
        else:
            c = 0.3  # Default from paper

        return apply_saguaro_rescaling(
            logits=logits,
            cached_token_indices=cached_token_indices,
            num_cached_tokens=self.fan_out,
            saguaro_c=c,
        )
