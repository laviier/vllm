# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace

import torch

from vllm.v1.spec_decode.draft_server import DraftServer


def test_eagle_shadow_lookup_matches_full_outcome_key():
    server = DraftServer.__new__(DraftServer)
    server.K = 2
    server.device = torch.device("cpu")
    server._eagle_shadow_keys = torch.tensor(
        [
            [10, 0, 100],
            [10, 1, 101],
            [11, 0, 200],
        ]
    )
    server._eagle_shadow_tokens = torch.tensor(
        [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
    )

    tokens, hits = server._lookup_eagle_shadow(
        seq_ids=torch.tensor([10, 10, 11]),
        k_accepted=torch.tensor([1, 1, 0]),
        bonus_tokens=torch.tensor([101, 999, 200]),
    )

    assert torch.equal(hits, torch.tensor([True, False, True]))
    assert torch.equal(tokens, torch.tensor([[3, 4], [0, 0], [5, 6]]))


def test_eagle_shadow_build_excludes_rejected_draft_tokens():
    server = DraftServer.__new__(DraftServer)
    server.K = 2
    server.device = torch.device("cpu")

    captured: dict[str, torch.Tensor] = {}

    def allocate(_self, **kwargs):
        captured["prefix_lens"] = kwargs["prefix_lens_override"].clone()
        n = kwargs["N"]
        return torch.zeros(n, 4, dtype=torch.int32), kwargs[
            "prefix_lens_override"
        ]

    server._allocate_branch_blocks_and_copy_kv = MethodType(allocate, server)

    def shadow_decode(**kwargs):
        captured["bonus_tokens"] = kwargs["bonus_tokens"].clone()
        captured["initial_hidden_states"] = kwargs[
            "initial_hidden_states"
        ].clone()
        return kwargs["bonus_tokens"].unsqueeze(1).expand(-1, 2).clone()

    runner = SimpleNamespace(
        recycle_dedicated_blocks=lambda _owner: None,
        eagle_shadow_tree_decode=shadow_decode,
    )
    logits = torch.zeros(1, 3, 12)
    logits[0, 0, 5] = 10
    logits[0, 0, 7] = 9
    logits[0, 1, 6] = 10
    logits[0, 1, 8] = 9
    logits[0, 2, 9] = 10
    feedback = torch.arange(6, dtype=torch.float32).view(1, 3, 2)

    server._build_eagle_shadow(
        runner=runner,
        seq_ids=torch.tensor([42]),
        draft_tokens=torch.tensor([[5, 6]]),
        outcome_logits=logits,
        feedback_trace=feedback,
        last_positions=torch.tensor([10]),
    )

    assert torch.equal(captured["bonus_tokens"], torch.tensor([7, 8, 9]))
    assert torch.equal(captured["prefix_lens"], torch.tensor([11, 12, 13]))
    assert torch.equal(captured["initial_hidden_states"], feedback[0])
    assert torch.equal(
        server._eagle_shadow_keys,
        torch.tensor([[42, 0, 7], [42, 1, 8], [42, 2, 9]]),
    )
    assert torch.equal(
        server._eagle_shadow_tokens,
        torch.tensor([[7, 7], [8, 8], [9, 9]]),
    )
