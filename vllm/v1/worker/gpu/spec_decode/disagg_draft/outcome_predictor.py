# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Verification Outcome Prediction and Geometric Fan-Out for disaggregated draft.

The outcome predictor determines the most likely verification outcomes
(k_accepted, bonus_token) that the target model will produce, and allocates
the "fan-out budget" F optimally across acceptance positions using the
geometric fan-out allocation from disagg draft paper Theorem 12.

Key insight: At each acceptance position k, the draft model's own logits
(excluding the token it actually sampled) provide a good prediction of
what the target's bonus token will be. We take the top-F_k tokens at each
position as bonus token candidates.

Reference: SSD paper §4.1, Theorem 12 (Optimal Budget Allocation)
"""

from __future__ import annotations

import math

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def compute_geometric_fanout(
    total_budget: int,
    num_positions: int,
    acceptance_rate: float = 0.85,
    power_law_exponent: float = 1.5,
) -> list[int]:
    """Compute optimal fan-out allocation using geometric series.

    Allocates the total fan-out budget across K+1 acceptance positions
    using the geometric series from Theorem 12. Earlier positions (more
    likely to be the acceptance point) get more fan-out.

    The allocation follows:
        F_k = F_0 * a^(k / (1 + r))  for k < K
        F_K = F_0 * a^(K / (1 + r)) * (1 - a)^(-1 / (1 + r))

    where:
        a = acceptance_rate (probability each token is accepted)
        r = power_law_exponent (controls how cache hit rate scales with F)
        F_0 = normalizing constant so sum(F_k) = total_budget

    Args:
        total_budget: Total number of fan-out entries to allocate.
            This is B_total = sum of all F_k across positions.
        num_positions: Number of acceptance positions (K + 1).
        acceptance_rate: Estimated per-token acceptance rate (a).
            POC measured: ~0.64-0.67 for Llama-1B→70B.
        power_law_exponent: Controls how 1-p_hit scales with F (r).
            Paper fits r ≈ 1.5 from calibration data.

    Returns:
        fan_out_list: List of length num_positions, where fan_out_list[k]
            is the number of bonus token candidates at position k.
            Guaranteed to sum to total_budget.
    """
    if num_positions <= 0:
        return []
    if num_positions == 1:
        return [total_budget]

    a = acceptance_rate
    r = power_law_exponent
    K = num_positions - 1  # last position index

    # Compute unnormalized weights
    # w_k = a^(k / (1+r)) for k < K
    # w_K = a^(K / (1+r)) * (1 - a)^(-1 / (1+r))
    exponent_factor = 1.0 / (1.0 + r)
    weights = []
    for k in range(K):
        w = a ** (k * exponent_factor)
        weights.append(w)

    # Last position weight (accounts for all tokens being accepted)
    w_K = a ** (K * exponent_factor)
    if a < 1.0:
        w_K *= (1.0 - a) ** (-exponent_factor)
    weights.append(w_K)

    # Normalize to sum to total_budget
    total_weight = sum(weights)
    if total_weight <= 0:
        # Fallback: uniform allocation
        base = total_budget // num_positions
        remainder = total_budget % num_positions
        return [base + (1 if i < remainder else 0) for i in range(num_positions)]

    fan_out_raw = [w / total_weight * total_budget for w in weights]

    if total_budget < num_positions:
        # Budget too small to give 1 to every position.
        # Give 1 to the top-budget positions by weight, 0 to the rest.
        fan_out = [0] * num_positions
        ranked = sorted(range(num_positions), key=lambda i: weights[i], reverse=True)
        for j in range(total_budget):
            fan_out[ranked[j]] = 1
        return fan_out

    # Round to integers while preserving the total (min=1 per position)
    fan_out = [max(1, int(math.floor(f))) for f in fan_out_raw]
    deficit = total_budget - sum(fan_out)

    # Distribute remaining budget to positions with largest fractional parts
    if deficit > 0:
        fractional_parts = [
            (fan_out_raw[i] - fan_out[i], i) for i in range(num_positions)
        ]
        fractional_parts.sort(reverse=True)
        for j in range(min(deficit, num_positions)):
            fan_out[fractional_parts[j][1]] += 1

    # Handle negative deficit (over-allocation from min=1 constraint)
    while sum(fan_out) > total_budget:
        # Remove from smallest allocation (preserve top positions)
        min_val = min(f for f in fan_out if f > 1)
        min_idx = fan_out.index(min_val)
        fan_out[min_idx] -= 1

    assert sum(fan_out) == total_budget, (
        f"Fan-out allocation error: sum={sum(fan_out)} != budget={total_budget}"
    )
    return fan_out


class OutcomePredictor:
    """Predicts verification outcomes and generates fan-out token candidates.

    For each acceptance position k ∈ [0, K], predicts the top-F_k most
    likely bonus tokens using the draft model's own logits. The sampled
    draft token at position k is excluded (since it becomes the "continuation"
    token, not the bonus token from rejection sampling).

    Args:
        num_speculative_tokens: K, speculation lookahead depth.
        total_fan_out: Total fan-out budget across all positions.
        acceptance_rate: Estimated acceptance rate for geometric allocation.
        power_law_exponent: Power-law exponent r for cache hit scaling.
        device: CUDA device.
    """

    def __init__(
        self,
        num_speculative_tokens: int,
        total_fan_out: int,
        acceptance_rate: float = 0.85,
        power_law_exponent: float = 1.5,
        device: torch.device = torch.device("cuda"),
    ):
        self.K = num_speculative_tokens
        self.total_fan_out = total_fan_out
        self.device = device

        # Compute per-position fan-out allocation
        num_positions = self.K + 1
        self.fan_out_list = compute_geometric_fanout(
            total_budget=total_fan_out,
            num_positions=num_positions,
            acceptance_rate=acceptance_rate,
            power_law_exponent=power_law_exponent,
        )
        self.fan_out_tensor = torch.tensor(
            self.fan_out_list, dtype=torch.int64, device=device
        )
        self.max_fan_out = max(self.fan_out_list)

        # Precomputed flat indexing tensors used by
        # ``_select_bonus_candidates`` every round. They depend only on
        # ``fan_out_list`` (fixed at init), so building them once at
        # startup eliminates ~14 small kernel launches per round.
        # Layout: for fan_out_list = [F_0, F_1, ..., F_K], the flats
        # are the concatenation of (k repeated F_k times) and
        # (arange(F_k)) for each k where F_k > 0.
        per_seq_k_chunks: list[torch.Tensor] = []
        per_seq_cand_chunks: list[torch.Tensor] = []
        for k, F_k in enumerate(self.fan_out_list):
            if F_k <= 0:
                continue
            per_seq_k_chunks.append(torch.full(
                (F_k,), k, dtype=torch.int64, device=device,
            ))
            per_seq_cand_chunks.append(torch.arange(
                F_k, dtype=torch.int64, device=device,
            ))
        empty = torch.zeros(0, dtype=torch.int64, device=device)
        self.per_seq_k_flat = (
            torch.cat(per_seq_k_chunks) if per_seq_k_chunks else empty
        )
        self.per_seq_cand_flat = (
            torch.cat(per_seq_cand_chunks) if per_seq_cand_chunks else empty
        )
        self.branches_per_seq = sum(self.fan_out_list)

        logger.info(
            "Disagg draft OutcomePredictor: K=%d, total_budget=%d, fan_out=%s",
            self.K,
            total_fan_out,
            self.fan_out_list,
        )

