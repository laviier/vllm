# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graph capture for decode_step and tree_decode_step.

Pre-records the model's forward pass at a curated set of batch sizes
so the runtime path can replay instead of relaunching kernels every
step. Two captures share one memory pool: ``_decode_graphs`` keyed by
batch size for ``decode_step``, ``_tree_graphs`` keyed by N (number of
branch tokens) for ``tree_decode_step``.

The mixin expects these attributes set by the consumer's ``__init__``:

    device, dtype, hidden_size, block_size, max_num_blocks,
    max_num_seqs, max_model_len, model, _draft_vllm_config,
    _decode_graphs (empty dict), _tree_graphs (empty dict),
    _decode_graphs_captured (False), _tree_decode_captured (False),
    _decode_graph_pool (None).

The mixin also expects ``_build_flash_attn_metadata`` and
``_build_slot_mapping_dict`` from ``DraftAttnMetadataMixin``.
"""

from __future__ import annotations

import torch

from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger

logger = init_logger(__name__)


class DraftCudaGraphMixin:
    """Mixin: CUDA graph capture for decode + tree decode."""

    def _capture_decode_graphs(self) -> None:
        """Capture CUDA graphs for decode_step at common batch sizes.

        With hybrid swap+JIT, decode graphs are used for JIT on cache
        misses (B_miss), which under N:M spec decode at high concurrency
        can land on any integer in [1..B_active]. Capture every size up
        to 16 individually so no B_miss value falls back to eager; use
        coarser steps above 16 where the B_miss distribution is less
        sensitive to exact match.
        """
        max_bs = min(self.max_num_seqs, 128)
        dense = list(range(1, 17))  # 1..16 — covers all B_miss at small B
        coarse = [24, 32, 48, 64, 96, 128]
        sizes = [bs for bs in dense + coarse if bs <= max_bs]
        if max_bs not in sizes:
            sizes.append(max_bs)
        logger.info("Capturing CUDA graphs for decode_step: bs=%s", sizes)
        self._capture_graphs_for_sizes(sizes, self._decode_graphs)
        self._decode_graphs_captured = True
        logger.info("CUDA graphs captured for %d decode sizes.", len(sizes))

    def _capture_tree_decode_graphs(self) -> None:
        """Capture CUDA graphs for tree_decode_step at common N values.

        Tree decode processes N branch tokens per step. The exact N
        depends on the call shape:

        - Sequential tree decode: N = B_total × sum(fan_out_list).
        - Parallel fanout: N = B_total × sum(fan_out_list) × K
          (one forward over all K depths, mask token at depth>0).
        - Parallel-fanout KV cleanup: N = B_total × sum(fan_out_list) ×
          (K-1) (one forward over depths 1..K-1, real tokens replacing
          the mask KVs).

        Tree-decode replay pads up to the next captured size, so a
        miss only costs the padding overhead — but wild mismatches
        (e.g. N=56 padded to 72) still incur ~30% extra compute. Dense
        capture below N=108 covers F=1/2/3 sequential at B≤8; coarser
        sizes from 126..504 cover sequential at higher concurrency
        and small parallel/cleanup calls. The 576..2160 range covers
        parallel-fanout and cleanup at multi-VS configs (e.g. 3V+1D
        at c=8 with K=5, fan_out=3 produces N=2160 for parallel and
        N=1728 for cleanup; without these captures both fall to eager).
        """
        dense = [7, 10, 14, 18, 21, 28, 35, 36, 42, 49, 54, 56,
                 63, 70, 72, 80, 84, 90, 98, 108]
        coarse = [126, 144, 168, 192, 256, 336, 504]
        # Parallel-fanout and KV-cleanup call shapes:
        #   parallel fanout call: N = B_total × sum_fan_out × K
        #   KV cleanup call:      N = B_total × sum_fan_out × (K-1)
        # Capture both for typical B_total values (per-VS B={4,8} ×
        # N_VS={1,2,3}). Sizes are computed from the live K and
        # sum_fan_out so they track config changes (different K or
        # disagg_fan_out) instead of being hardcoded for one config.
        K = self._num_spec_tokens
        sum_fan = self._sum_fan_out
        parallel: set[int] = set()
        # B_total = per-VS batch × N_VS. Cover per-VS up to 8 across
        # 1..4 connected verifiers (B_total up to 32). Larger merged
        # configs fall back to eager — fine since they're far from
        # the typical deployment sweet spot of 1-3V at c=8.
        for b_total in (4, 8, 12, 16, 24, 32):
            n_branches = b_total * sum_fan
            for multiplier in (K - 1, K):
                if multiplier <= 0:
                    continue
                parallel.add(n_branches * multiplier)
        # Deferred-cleanup call shape: H_hit × (K - 1), where H_hit ≤
        # B_total. Capture every B_total × (K - 1) so a hit-saturated
        # round at any concurrency hits a graph at the exact requested
        # size — padding to a larger captured size produced FP-
        # nondeterminism (different kernel grid → different reduction
        # order → ~1 % AL drift on mt-bench).
        if K > 1:
            for b_total in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32):
                parallel.add(b_total * (K - 1))
        # Fused cleanup+glue call shape: H × K + (B_total − H), where
        # H is the hit count this round. Land at exact captured sizes
        # for the all-hit (H=B → B × K) and all-miss (H=0 → B) extremes
        # plus typical mixed shapes. Reuses dense+coarse for small
        # values; adds B × K for B in {1..8, 12, 16, 24, 32}.
        for b_total in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32):
            parallel.add(b_total)              # all-miss
            parallel.add(b_total * K)          # all-hit
        sizes = sorted(set(dense + coarse) | parallel)
        logger.info("Capturing CUDA graphs for tree_decode_step: N=%s", sizes)
        self._capture_graphs_for_sizes(sizes, self._tree_graphs)
        self._tree_decode_captured = True
        logger.info("CUDA graphs captured for %d tree sizes.", len(sizes))

    def _capture_graphs_for_sizes(
        self,
        sizes: list[int],
        target_dict: dict[int, dict],
    ) -> None:
        """Shared CUDA graph capture logic for decode and tree decode.

        Pre-allocates input/output tensors at the max size, then captures
        a graph for each size. All graphs share the same memory pool.

        Args:
            sizes: List of token counts to capture graphs for.
            target_dict: Dict to store captured graphs (keyed by size).
        """
        max_n = max(sizes)
        g_input_ids = torch.zeros(max_n, dtype=torch.int64, device=self.device)
        g_positions = torch.zeros(max_n, dtype=torch.long, device=self.device)
        g_slot_mapping = torch.zeros(
            max_n, dtype=torch.int64, device=self.device
        )
        g_seq_lens = torch.ones(max_n, dtype=torch.int32, device=self.device)
        g_block_tables = torch.zeros(
            max_n, self.max_num_blocks,
            dtype=torch.int32, device=self.device,
        )
        g_query_start_loc = torch.arange(
            max_n + 1, dtype=torch.int32, device=self.device
        )
        g_hidden = torch.zeros(
            max_n, self.hidden_size, dtype=self.dtype, device=self.device
        )

        for n in reversed(sizes):
            attn_metadata = self._build_flash_attn_metadata(
                num_tokens=n,
                seq_lens_tensor=g_seq_lens[:n],
                max_seq_len=self.max_model_len,
                max_query_len=1,
                query_start_loc=g_query_start_loc[:n + 1],
                block_table=g_block_tables[:n],
                slot_mapping=g_slot_mapping[:n],
            )
            slot_mapping_dict = self._build_slot_mapping_dict(
                g_slot_mapping[:n]
            )
            batch_descriptor = BatchDescriptor(num_tokens=n)

            # Warmup
            with set_forward_context(
                attn_metadata=attn_metadata,
                vllm_config=self._draft_vllm_config,
                num_tokens=n,
                slot_mapping=slot_mapping_dict,
                batch_descriptor=batch_descriptor,
            ):
                g_hidden[:n] = self.model(
                    input_ids=g_input_ids[:n],
                    positions=g_positions[:n],
                )

            # Capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._decode_graph_pool):
                with set_forward_context(
                    attn_metadata=attn_metadata,
                    vllm_config=self._draft_vllm_config,
                    num_tokens=n,
                    slot_mapping=slot_mapping_dict,
                    batch_descriptor=batch_descriptor,
                ):
                    g_hidden[:n] = self.model(
                        input_ids=g_input_ids[:n],
                        positions=g_positions[:n],
                    )

            if self._decode_graph_pool is None:
                self._decode_graph_pool = graph.pool()

            target_dict[n] = {
                "graph": graph,
                "input_ids": g_input_ids,
                "positions": g_positions,
                "slot_mapping": g_slot_mapping,
                "seq_lens": g_seq_lens,
                "block_tables": g_block_tables,
                "query_start_loc": g_query_start_loc,
                "hidden": g_hidden,
                "attn_metadata": attn_metadata,
                "slot_mapping_dict": slot_mapping_dict,
                "batch_descriptor": batch_descriptor,
            }
            torch.cuda.synchronize()
