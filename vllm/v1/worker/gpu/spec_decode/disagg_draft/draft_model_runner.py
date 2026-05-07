# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Draft Model Runner for disagg draft Disaggregated Draft Worker.

Manages the draft model lifecycle: loading, KV cache allocation,
prefill, and sequential decode forward passes. Runs on the draft GPU
independently from the target model's TP group.

This is intentionally simpler than vLLM's full ModelRunner (MRV2) since
the draft model:
- Runs on a single GPU (TP=1)
- Doesn't need async scheduling
- Doesn't need CUDA graph capture for Phase 1 (added in Phase 2)
- Has its own KV cache and block manager

The draft model runner provides two key operations:
1. prefill(): Process prefix tokens for new sequences
2. decode_step(): Generate one token given input_id + position

Reference: SSD ref impl ssd/engine/model_runner.py
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.disagg_draft.kv_cache_manager import (
    DraftKVCacheMixin,
)

logger = init_logger(__name__)


class DraftModelRunner(DraftKVCacheMixin):
    """Manages the draft model for disagg draft disaggregated speculation.

    Handles model loading, KV cache allocation, and forward passes
    for the standalone draft model running on a separate GPU.

    This is a simplified model runner that doesn't use the full MRV2
    infrastructure. It's designed for single-GPU, single-model operation
    with minimal overhead.

    Args:
        vllm_config: Full vLLM configuration (draft model config used).
        device: CUDA device for the draft model.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.device = device

        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        self.draft_config = spec_config.draft_model_config
        assert self.draft_config is not None

        self.vocab_size = self.draft_config.get_vocab_size()
        self.hidden_size = self.draft_config.get_hidden_size()
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = self.draft_config.max_model_len
        self._num_spec_tokens = spec_config.num_speculative_tokens

        # KV cache parameters
        self.block_size = vllm_config.cache_config.block_size
        self.num_kv_heads = getattr(
            self.draft_config.hf_text_config, "num_key_value_heads", 1
        )
        self.num_attn_heads = getattr(
            self.draft_config.hf_text_config, "num_attention_heads", 1
        )
        self.num_layers = self.draft_config.hf_text_config.num_hidden_layers
        self.head_dim = getattr(
            self.draft_config.hf_text_config, "head_dim", 128
        )

        # Model and KV cache (allocated during load_model)
        self.model: nn.Module | None = None
        self.kv_caches: list[torch.Tensor] | None = None
        self.num_kv_blocks: int = 0
        self._model_loaded = False

        # Block table: maps seq_id → [physical_blocks]
        self.max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_blocks = (
            self.max_model_len + self.block_size - 1
        ) // self.block_size

        # Block allocator for the draft GPU.
        # Uses a free-list backed by a bump pointer: freed blocks go onto
        # _free_list and are reused before bumping _next_free_block.
        self._next_free_block = 0
        self._free_list: list[int] = []  # recycled block IDs
        self._block_tables: dict[int, list[int]] = {}
        # Dedicated blocks reserved by the last _build_next_cache call,
        # partitioned per verify server. Each VS's entry is recycled
        # only at the start of *that VS's* next cache build, so
        # preserved cache entries from other VSes are not invalidated
        # when a peer VS rebuilds its cache. Under N:M with several
        # VSes sharing a draft this is what keeps cross-VS cache
        # entries pointing at valid KV data.
        self._dedicated_blocks_by_vs: dict[str, list[int]] = {}
        # Track sequence lengths for decode positioning
        self._seq_lens: dict[int, int] = {}

        # GPU-resident block table for vectorized slot mapping.
        # Shape: [max_num_seqs, max_num_blocks]. Updated when blocks
        # are allocated. Avoids Python dict lookups in decode_step.
        self._block_table_gpu = torch.zeros(
            self.max_num_seqs + 1024,  # extra room for seq_ids
            self.max_num_blocks,
            dtype=torch.int32,
            device=device,
        )
        # Pre-allocated tensors for decode_step (avoid per-call alloc)
        self._decode_query_start_loc = torch.arange(
            self.max_num_seqs + 1, dtype=torch.int32, device=device
        )

        # KV cache snapshot for tree decode branching
        self._kv_snapshot: dict[int, int] | None = None

        # Pre-allocated buffers for tree decode (avoid per-call allocation)
        self._tree_decode_captured = False
        self._tree_graphs: dict[int, tuple] = {}  # N → (graph, buffers)

        # CUDA graph state for decode_step
        self._decode_graphs: dict[int, dict] = {}  # bs → graph + buffers
        self._decode_graph_pool = None
        self._decode_graphs_captured = False

    def glue_decode(
        self,
        tokens: torch.Tensor,
        seq_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run a 'glue' decode step: feed one token per sequence.

        Args:
            tokens: [B] — the recovery tokens.
            seq_ids: [B] — sequence IDs.

        Returns:
            logits: [B, V] — logits for the next position.
        """
        positions = torch.tensor(
            [self._seq_lens.get(int(sid), 0) for sid in seq_ids.tolist()],
            dtype=torch.long,
            device=self.device,
        )
        logits, _ = self.decode_step(tokens, positions, seq_ids)
        return logits

    def glue_decode_batched(
        self,
        input_ids: torch.Tensor,
        seq_ids: torch.Tensor,
        tokens_per_seq: int,
    ) -> torch.Tensor:
        """Run a batched glue decode: K+1 tokens per sequence in one pass.

        Feeds recovery + K draft tokens in a single forward pass using
        FlashAttention varlen batching. Each sequence has `tokens_per_seq`
        query tokens. Returns logits at all positions.

        Args:
            input_ids: [B * tokens_per_seq] — flattened tokens
                (recovery + K draft tokens per sequence).
            seq_ids: [B] — sequence IDs.
            tokens_per_seq: Number of query tokens per sequence (K+1).

        Returns:
            logits: [B * tokens_per_seq, V] — logits at each position.
        """
        assert self._model_loaded
        B = seq_ids.shape[0]
        total = B * tokens_per_seq
        seq_ids_list = seq_ids.tolist()

        # Build positions: each sequence starts at its current _seq_lens
        positions = torch.zeros(total, dtype=torch.long, device=self.device)
        expanded_seq_ids = []
        for i, sid in enumerate(seq_ids_list):
            start_pos = self._seq_lens.get(int(sid), 0)
            offset = i * tokens_per_seq
            positions[offset:offset + tokens_per_seq] = (
                torch.arange(tokens_per_seq, device=self.device) + start_pos
            )
            expanded_seq_ids.extend([int(sid)] * tokens_per_seq)

        slot_mapping = self._compute_slot_mapping(positions, expanded_seq_ids)
        block_tables = self._get_block_table_tensor(seq_ids)

        # seq_lens for attention: each seq attends to start_pos + tokens_per_seq
        seq_lens_list = [
            self._seq_lens.get(int(sid), 0) + tokens_per_seq
            for sid in seq_ids_list
        ]
        seq_lens_t = torch.tensor(
            seq_lens_list, dtype=torch.int32, device=self.device
        )
        max_seq_len = int(seq_lens_t.max().item())

        query_start_loc = torch.zeros(
            B + 1, dtype=torch.int32, device=self.device
        )
        query_start_loc[1:] = torch.arange(
            1, B + 1, device=self.device, dtype=torch.int32
        ) * tokens_per_seq

        attn_metadata = self._build_flash_attn_metadata(
            num_tokens=total,
            seq_lens_tensor=seq_lens_t,
            max_seq_len=max_seq_len,
            max_query_len=tokens_per_seq,
            query_start_loc=query_start_loc,
            block_table=block_tables,
            slot_mapping=slot_mapping,
        )

        slot_mapping_dict = self._build_slot_mapping_dict(slot_mapping)
        batch_descriptor = BatchDescriptor(num_tokens=total)
        with set_forward_context(
            attn_metadata=attn_metadata,
            vllm_config=self._draft_vllm_config,
            num_tokens=total,
            slot_mapping=slot_mapping_dict,
            batch_descriptor=batch_descriptor,
        ):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
            )

        if hasattr(self.model, "compute_logits"):
            logits = self.model.compute_logits(hidden_states)
        elif hasattr(self.model, "lm_head"):
            logits = self.model.lm_head(hidden_states)
        else:
            logits = torch.matmul(
                hidden_states,
                self.model.get_input_embeddings().weight.T,
            )

        # Update _seq_lens
        for sid in seq_ids_list:
            self._seq_lens[int(sid)] = (
                self._seq_lens.get(int(sid), 0) + tokens_per_seq
            )

        return logits[:, :self.vocab_size]

    def tree_decode_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_ids_expanded: torch.Tensor,
        block_tables: torch.Tensor,
        max_seq_len_hint: int | None = None,
    ) -> torch.Tensor:
        """Run one tree decode step, using CUDA graphs when available.

        Each branch token is a separate 1-token "sequence" in the
        FlashAttention varlen batch. Uses captured CUDA graphs for
        common N values, padding to the nearest captured size.

        Returns:
            logits: [N, V] — logits for each branch.
        """
        assert self._model_loaded
        N = input_ids.shape[0]

        # Compute slot mapping from per-branch block tables
        logical_blocks = (positions // self.block_size).to(torch.int64)
        offsets = (positions % self.block_size).to(torch.int64)
        physical_blocks = block_tables[
            torch.arange(N, device=self.device), logical_blocks
        ].to(torch.int64)
        slot_mapping = physical_blocks * self.block_size + offsets

        seq_lens_i32 = seq_lens.to(torch.int32)
        if max_seq_len_hint is not None:
            max_seq_len = max_seq_len_hint
        else:
            max_seq_len = int(seq_lens_i32.max().item())

        # Find matching CUDA graph (exact N or next larger captured size)
        graph_n = None
        if self._tree_decode_captured:
            for candidate in sorted(self._tree_graphs.keys()):
                if candidate >= N:
                    graph_n = candidate
                    break

        if graph_n is not None:
            g = self._tree_graphs[graph_n]
            # Copy inputs into graph's pre-allocated buffers (pad to graph_n)
            g["input_ids"][:N].copy_(input_ids)
            g["positions"][:N].copy_(positions)
            g["slot_mapping"][:N].copy_(slot_mapping)
            g["seq_lens"][:N].copy_(seq_lens_i32)
            # Pad extra slots with safe values (seq_len=1, slot=0)
            if graph_n > N:
                g["input_ids"][N:graph_n].zero_()
                g["positions"][N:graph_n].zero_()
                g["slot_mapping"][N:graph_n].zero_()
                g["seq_lens"][N:graph_n].fill_(1)
            g["block_tables"][:N].copy_(block_tables[:N])
            if graph_n > N:
                g["block_tables"][N:graph_n].zero_()
            g["attn_metadata"].max_seq_len = max_seq_len
            for layer_name in g["slot_mapping_dict"]:
                g["slot_mapping_dict"][layer_name] = g["slot_mapping"][:graph_n]
            # Replay
            g["graph"].replay()
            hidden_states = g["hidden"][:N]
        else:
            # Eager fallback
            query_start_loc = torch.arange(
                N + 1, dtype=torch.int32, device=self.device
            )
            attn_metadata = self._build_flash_attn_metadata(
                num_tokens=N,
                seq_lens_tensor=seq_lens_i32,
                max_seq_len=max_seq_len,
                max_query_len=1,
                query_start_loc=query_start_loc,
                block_table=block_tables,
                slot_mapping=slot_mapping,
            )
            slot_mapping_dict = self._build_slot_mapping_dict(slot_mapping)
            batch_descriptor = BatchDescriptor(num_tokens=N)
            with set_forward_context(
                attn_metadata=attn_metadata,
                vllm_config=self._draft_vllm_config,
                num_tokens=N,
                slot_mapping=slot_mapping_dict,
                batch_descriptor=batch_descriptor,
            ):
                hidden_states = self.model(
                    input_ids=input_ids,
                    positions=positions,
                )

        # Compute logits
        if hasattr(self.model, "compute_logits"):
            logits = self.model.compute_logits(hidden_states)
        elif hasattr(self.model, "lm_head"):
            logits = self.model.lm_head(hidden_states)
        else:
            logits = torch.matmul(
                hidden_states,
                self.model.get_input_embeddings().weight.T,
            )

        return logits[:, :self.vocab_size]

    def load_model(self) -> None:
        """Load the draft model and allocate KV cache."""
        logger.info(
            "Loading disagg draft draft model: %s on %s",
            self.draft_config.model,
            self.device,
        )

        t0 = time.perf_counter()

        from copy import deepcopy
        from vllm.config.compilation import (
            CompilationConfig, CompilationMode, CUDAGraphMode,
        )
        from vllm.model_executor.model_loader import get_model

        # Create a modified vllm_config with the draft model config
        # substituted for model_config. This ensures the model loader
        # uses the draft model's hidden_size, num_layers, etc. instead
        # of the target model's.
        # Also disable torch.compile and CUDA graphs for the draft model
        # since our simplified runner doesn't provide the full forward
        # context that the compiled/captured graph expects.
        draft_vllm_config = deepcopy(self.vllm_config)
        draft_vllm_config.model_config = self.draft_config
        draft_vllm_config.compilation_config = CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
            custom_ops=["all"],
        )
        # Use TP=1 parallel config for the draft model.
        spec_config = self.vllm_config.speculative_config
        if (spec_config is not None
                and spec_config.draft_parallel_config is not None):
            draft_vllm_config.parallel_config = spec_config.draft_parallel_config
        else:
            draft_vllm_config.parallel_config = deepcopy(
                self.vllm_config.parallel_config
            )
            draft_vllm_config.parallel_config.tensor_parallel_size = 1
            draft_vllm_config.parallel_config.pipeline_parallel_size = 1

        self.model = get_model(
            vllm_config=draft_vllm_config,
            model_config=self.draft_config,
        )
        self.model.eval()

        # Store the draft vllm_config — attention layers register
        # themselves in compilation_config.static_forward_context
        # during model construction, so we must use this config
        # (not the original) for set_forward_context() calls.
        self._draft_vllm_config = draft_vllm_config

        dt = time.perf_counter() - t0
        logger.info("Draft model loaded in %.1f seconds.", dt)

        self._allocate_kv_cache()
        self._bind_kv_cache_to_attention_layers()
        self._model_loaded = True

        try:
            self._capture_decode_graphs()
        except Exception as e:
            logger.warning(
                "CUDA graph capture failed: %s. Using eager decode.", e
            )
            self._decode_graphs_captured = False

        try:
            self._capture_tree_decode_graphs()
        except Exception as e:
            logger.warning(
                "Tree decode CUDA graph capture failed: %s. Using eager.", e
            )
            self._tree_decode_captured = False

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

        Tree decode processes N = B × sum(fan_out_list) branch tokens
        per step. The exact N depends on both B and the geometric
        fan-out allocation, so capture a dense set of small sizes
        (covering F=1/2/3 at B<=8) plus coarser larger sizes for
        high-concurrency cases. Tree-decode replay pads up to the
        next captured size, so a miss only costs the padding overhead
        — but wild mismatches (e.g. N=56 padded to 72) still incur
        ~30% extra compute, and at F=1 we hit them often enough to
        regress measurably. Dense capture below N=100 avoids this.
        """
        dense = [7, 10, 14, 18, 21, 28, 35, 36, 42, 49, 54, 56,
                 63, 70, 72, 80, 84, 90, 98, 108]
        coarse = [126, 144, 168, 192, 256, 336, 504]
        sizes = sorted(set(dense + coarse))
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

    def _get_block_table_tensor(
        self, seq_ids: torch.Tensor | list[int]
    ) -> torch.Tensor:
        """Build a [B, max_blocks] block table tensor from GPU-resident table."""
        if isinstance(seq_ids, list):
            seq_ids_t = torch.tensor(
                seq_ids, dtype=torch.int64, device=self.device
            )
        else:
            seq_ids_t = seq_ids.to(torch.int64)
        # Return a contiguous copy so callers cannot mutate _block_table_gpu
        return self._block_table_gpu[seq_ids_t].contiguous()

    def _compute_slot_mapping(
        self,
        positions: torch.Tensor,
        seq_ids: torch.Tensor | list[int],
    ) -> torch.Tensor:
        """Compute slot mapping using GPU-resident block table (vectorized).

        For each token, the physical slot is:
            physical_block * block_size + offset_in_block
        where:
            logical_block = position // block_size
            offset_in_block = position % block_size
            physical_block = block_table_gpu[seq_id, logical_block]
        """
        if isinstance(seq_ids, list):
            seq_ids_t = torch.tensor(
                seq_ids, dtype=torch.int64, device=self.device
            )
        else:
            seq_ids_t = seq_ids.to(torch.int64)
        logical_blocks = (positions // self.block_size).to(torch.int64)
        offsets = (positions % self.block_size).to(torch.int64)
        physical_blocks = self._block_table_gpu[
            seq_ids_t, logical_blocks
        ].to(torch.int64)
        return physical_blocks * self.block_size + offsets

    def _build_slot_mapping_dict(
        self, slot_mapping: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Build a slot_mapping dict keyed by attention layer names.

        In V1, `unified_kv_cache_update` looks up slot_mapping by layer
        name: `forward_context.slot_mapping.get(layer_name)`.  If the
        key doesn't match, the lookup returns None and the KV cache is
        NOT updated.  We must populate the dict with every attention
        layer's registered name so each layer gets the slot mapping.
        """
        from vllm.model_executor.layers.attention import Attention

        forward_ctx = (
            self._draft_vllm_config.compilation_config.static_forward_context
        )
        mapping: dict[str, torch.Tensor] = {}
        for layer_name, layer in forward_ctx.items():
            if isinstance(layer, Attention):
                mapping[layer_name] = slot_mapping
        return mapping

    def _build_flash_attn_metadata(
        self,
        num_tokens: int,
        seq_lens_tensor: torch.Tensor,
        max_seq_len: int,
        max_query_len: int,
        query_start_loc: torch.Tensor,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """Build FlashAttentionMetadata for forward passes.

        Constructs the minimal metadata needed by the FlashAttention
        backend to perform paged attention with our KV cache.
        """
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

        return FlashAttentionMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens_tensor,
            block_table=block_table,
            slot_mapping=slot_mapping,
            # Cascade attention disabled for draft model
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
            # FA scheduling metadata (None = use FA heuristics)
            scheduler_metadata=None,
            prefix_scheduler_metadata=None,
        )

    # ---------------------------------------------------------------
    # Forward passes
    # ---------------------------------------------------------------

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        num_tokens_per_seq: torch.Tensor,
        seq_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run prefill forward pass for new sequences.

        Processes the prefix tokens to populate the draft model's KV cache.
        Returns the logits at the last position of each sequence.

        Args:
            input_ids: [total_tokens] — flattened input token IDs.
            num_tokens_per_seq: [B] — per-sequence token counts.
            seq_ids: [B] — sequence IDs for block allocation.

        Returns:
            last_logits: [B, V] — logits at last position per sequence.
        """
        assert self._model_loaded, "Call load_model() first"
        B = num_tokens_per_seq.shape[0]
        total = input_ids.shape[0]

        # Allocate blocks for prompt + initial headroom.
        # Additional blocks are grown on-demand via ensure_blocks()
        # during decode, so we only need a small buffer here.
        initial_headroom = 256  # tokens
        for i in range(B):
            n = int(num_tokens_per_seq[i].item())
            sid = int(seq_ids[i].item())
            self.allocate_blocks(sid, n + initial_headroom)
            self._seq_lens[sid] = n

        # Build positions: [total_tokens]
        positions = torch.zeros(total, dtype=torch.long, device=self.device)
        offset = 0
        expanded_seq_ids = []
        for i in range(B):
            n = int(num_tokens_per_seq[i].item())
            positions[offset:offset + n] = torch.arange(n, device=self.device)
            expanded_seq_ids.extend([int(seq_ids[i].item())] * n)
            offset += n

        # Compute slot mapping (vectorized via GPU-resident block table)
        slot_mapping = self._compute_slot_mapping(positions, expanded_seq_ids)

        # Build block table from GPU-resident table
        block_tables = self._get_block_table_tensor(seq_ids)

        # Build FlashAttention metadata for prefill.
        max_prompt_len = int(num_tokens_per_seq.max().item())

        # query_start_loc: cumulative sum of per-seq token counts
        query_start_loc = torch.zeros(
            B + 1, dtype=torch.int32, device=self.device
        )
        torch.cumsum(
            num_tokens_per_seq.to(torch.int32), dim=0, out=query_start_loc[1:]
        )

        attn_metadata = self._build_flash_attn_metadata(
            num_tokens=total,
            seq_lens_tensor=num_tokens_per_seq.to(torch.int32),
            max_seq_len=max_prompt_len,
            max_query_len=max_prompt_len,
            query_start_loc=query_start_loc,
            block_table=block_tables,
            slot_mapping=slot_mapping,
        )

        # Build slot mapping dict keyed by layer name so each attention
        # layer's KV cache gets updated via unified_kv_cache_update.
        slot_mapping_dict = self._build_slot_mapping_dict(slot_mapping)

        # Run model forward with proper context
        batch_descriptor = BatchDescriptor(num_tokens=total)
        with set_forward_context(
            attn_metadata=attn_metadata,
            vllm_config=self._draft_vllm_config,
            num_tokens=total,
            slot_mapping=slot_mapping_dict,
            batch_descriptor=batch_descriptor,
        ):
            # V1 models don't accept kv_caches as a forward argument;
            # KV cache is managed through the attention backend via
            # set_forward_context.
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
            )

        # Extract logits at the last position of each sequence
        last_indices = torch.cumsum(num_tokens_per_seq, dim=0) - 1
        last_hidden = hidden_states[last_indices]

        # Compute logits using the model's lm_head
        if hasattr(self.model, "compute_logits"):
            last_logits = self.model.compute_logits(last_hidden)
        elif hasattr(self.model, "lm_head"):
            last_logits = self.model.lm_head(last_hidden)
        else:
            # Fallback: use the model's output projection
            last_logits = torch.matmul(
                last_hidden,
                self.model.get_input_embeddings().weight.T,
            )

        logger.debug("Draft prefill: %d sequences, %d tokens", B, total)
        return last_logits[:, :self.vocab_size]

    @torch.inference_mode()
    def decode_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a single decode step, using CUDA graphs when available."""
        assert self._model_loaded, "Call load_model() first"
        B = input_ids.shape[0]

        # Ensure enough blocks for current positions (needed for
        # standalone draft model which doesn't pre-allocate via
        # _handle_speculation's ensure_blocks call).
        for i, sid in enumerate(seq_ids.tolist()):
            self.ensure_blocks(int(sid), int(positions[i].item()) + 1)

        # Vectorized slot mapping (GPU, no Python loops)
        logical_blocks = (positions // self.block_size).to(torch.int64)
        offsets = (positions % self.block_size).to(torch.int64)
        seq_ids_long = seq_ids.to(torch.int64)
        physical_blocks = self._block_table_gpu[
            seq_ids_long, logical_blocks
        ].to(torch.int64)
        slot_mapping = physical_blocks * self.block_size + offsets

        block_tables = self._block_table_gpu[seq_ids_long]

        seq_lens = (positions + 1).to(torch.int32)
        max_seq_len = int(seq_lens.max().item())

        # Try CUDA graph replay
        if self._decode_graphs_captured and B in self._decode_graphs:
            g = self._decode_graphs[B]
            # Copy inputs into graph's pre-allocated buffers
            g["input_ids"][:B].copy_(input_ids)
            g["positions"][:B].copy_(positions)
            g["slot_mapping"][:B].copy_(slot_mapping)
            g["seq_lens"][:B].copy_(seq_lens)
            g["block_tables"][:B].copy_(block_tables)
            # Update slot_mapping_dict references
            for layer_name in g["slot_mapping_dict"]:
                g["slot_mapping_dict"][layer_name] = g["slot_mapping"][:B]
            # Replay graph
            g["graph"].replay()
            hidden_states = g["hidden"][:B]
        else:
            # Eager fallback for batch sizes without captured graphs
            query_start_loc = self._decode_query_start_loc[:B + 1]
            attn_metadata = self._build_flash_attn_metadata(
                num_tokens=B,
                seq_lens_tensor=seq_lens,
                max_seq_len=max_seq_len,
                max_query_len=1,
                query_start_loc=query_start_loc,
                block_table=block_tables,
                slot_mapping=slot_mapping,
            )
            slot_mapping_dict = self._build_slot_mapping_dict(slot_mapping)
            batch_descriptor = BatchDescriptor(num_tokens=B)
            with set_forward_context(
                attn_metadata=attn_metadata,
                vllm_config=self._draft_vllm_config,
                num_tokens=B,
                slot_mapping=slot_mapping_dict,
                batch_descriptor=batch_descriptor,
            ):
                hidden_states = self.model(
                    input_ids=input_ids,
                    positions=positions,
                )

        # Compute logits
        if hasattr(self.model, "compute_logits"):
            logits = self.model.compute_logits(hidden_states)
        elif hasattr(self.model, "lm_head"):
            logits = self.model.lm_head(hidden_states)
        else:
            logits = torch.matmul(
                hidden_states,
                self.model.get_input_embeddings().weight.T,
            )

        # Update tracked sequence lengths
        seq_ids_list = seq_ids.tolist()
        for i, sid in enumerate(seq_ids_list):
            self._seq_lens[int(sid)] = int(positions[i].item()) + 1

        return logits[:, :self.vocab_size], hidden_states

    @torch.inference_mode()
    def sequential_speculate(
        self,
        recovery_tokens: torch.Tensor,
        positions: torch.Tensor,
        seq_ids: torch.Tensor,
        num_steps: int,
        temperature: torch.Tensor | None = None,
        saguaro_sampler=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run K sequential decode steps for JIT fallback speculation.

        Args:
            recovery_tokens: [B] — starting token (bonus token from verification).
            positions: [B] — starting position in each sequence.
            seq_ids: [B] — sequence IDs.
            num_steps: K — number of tokens to generate.
            temperature: [B] — sampling temperatures (None = greedy).
            saguaro_sampler: Optional SaguaroSampler to apply before sampling.
                Suppresses top-F token probabilities to increase cache hit rate.

        Returns:
            draft_tokens: [B, K] — generated draft tokens.
            draft_logits: [B, K, V] — logits at each step (before Saguaro).
        """
        B = recovery_tokens.shape[0]
        V = self.vocab_size

        draft_tokens = torch.zeros(
            B, num_steps, dtype=torch.int64, device=self.device
        )
        draft_logits = torch.zeros(
            B, num_steps, V, dtype=self.dtype, device=self.device
        )

        current_ids = recovery_tokens
        current_pos = positions.clone()

        for step in range(num_steps):
            logits, _ = self.decode_step(current_ids, current_pos, seq_ids)

            # Store original logits (before Saguaro) for rejection sampling
            draft_logits[:, step] = logits

            # Apply Saguaro rescaling before sampling to increase cache hit rate
            sample_logits = logits
            if saguaro_sampler is not None:
                sample_logits = saguaro_sampler.apply(
                    logits, temperature=temperature,
                )

            # Sample next token — always use argmax (greedy).
            # The disaggregated draft can't match the target's random
            # state (Gumbel noise seeds), so stochastic sampling would
            # produce different tokens even from identical distributions.
            # Argmax maximizes the chance of matching the target's top-1
            # prediction. Temperature-based acceptance is handled by the
            # target's rejection sampler using draft_probs.
            next_tokens = sample_logits.argmax(dim=-1)

            draft_tokens[:, step] = next_tokens
            current_ids = next_tokens
            current_pos = current_pos + 1

        return draft_tokens, draft_logits

    def parallel_speculate(
        self,
        recovery_tokens: torch.Tensor,
        positions: torch.Tensor,
        seq_ids: torch.Tensor,
        num_steps: int,
        mask_token_id: int,
        temperature: torch.Tensor | None = None,
        saguaro_sampler=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run parallel drafting: predict K tokens in a single forward pass.

        Instead of K sequential decode steps, builds an input of
        [bonus_token, <mask>, <mask>, ..., <mask>] per sequence and
        runs one prefill-style forward pass. The model must be trained
        to predict tokens at <mask> positions.

        Args:
            recovery_tokens: [B] — starting token (bonus token).
            positions: [B] — starting position in each sequence.
            seq_ids: [B] — sequence IDs.
            num_steps: K — number of draft tokens to generate.
            mask_token_id: Token ID for <mask> in the model's vocabulary.
            temperature: [B] — sampling temperatures (None = greedy).
            saguaro_sampler: Optional SaguaroSampler for cache hit rate.

        Returns:
            draft_tokens: [B, K] — generated draft tokens.
            draft_logits: [B, K, V] — logits at each position.
        """
        assert self._model_loaded, "Call load_model() first"
        B = recovery_tokens.shape[0]
        K = num_steps
        V = self.vocab_size
        Kp1 = K + 1  # bonus_token + K mask tokens

        # Build input: [bonus_token, <mask>, <mask>, ..., <mask>] per seq
        # Total tokens: B * (K+1)
        total_tokens = B * Kp1
        input_ids = torch.full(
            (total_tokens,), mask_token_id,
            dtype=torch.int64, device=self.device,
        )
        # Set the first token of each sequence to the bonus token
        for i in range(B):
            input_ids[i * Kp1] = recovery_tokens[i]

        # Build positions: [pos, pos+1, pos+2, ..., pos+K] per seq
        all_positions = torch.zeros(
            total_tokens, dtype=torch.int64, device=self.device,
        )
        for i in range(B):
            base_pos = positions[i].item()
            all_positions[i * Kp1 : (i + 1) * Kp1] = (
                torch.arange(Kp1, device=self.device) + base_pos
            )

        # Ensure blocks are allocated for all positions
        for i, sid in enumerate(seq_ids.tolist()):
            max_pos = int(positions[i].item()) + K
            self.ensure_blocks(int(sid), max_pos + 1)

        # Build slot mapping for all tokens
        expanded_seq_ids: list[int] = []
        for i in range(B):
            expanded_seq_ids.extend([int(seq_ids[i].item())] * Kp1)
        slot_mapping = self._compute_slot_mapping(
            all_positions, expanded_seq_ids,
        )

        # Build block tables
        block_tables = self._get_block_table_tensor(seq_ids)

        # Build prefill-style attention metadata
        # Each sequence has Kp1 query tokens
        seqlens_q = torch.full(
            (B,), Kp1, dtype=torch.int32, device=self.device,
        )
        query_start_loc = torch.zeros(
            B + 1, dtype=torch.int32, device=self.device,
        )
        torch.cumsum(seqlens_q, dim=0, out=query_start_loc[1:])

        # Total context length per sequence = existing KV + new tokens
        seq_lens_list = []
        for i in range(B):
            base_pos = int(positions[i].item())
            seq_lens_list.append(base_pos + Kp1)
        seq_lens_t = torch.tensor(
            seq_lens_list, dtype=torch.int32, device=self.device,
        )
        max_seq_len = int(seq_lens_t.max().item())

        from vllm.forward_context import BatchDescriptor, set_forward_context

        attn_metadata = self._build_flash_attn_metadata(
            num_tokens=total_tokens,
            seq_lens_tensor=seq_lens_t,
            max_seq_len=max_seq_len,
            max_query_len=Kp1,
            query_start_loc=query_start_loc,
            block_table=block_tables,
            slot_mapping=slot_mapping,
        )
        slot_mapping_dict = self._build_slot_mapping_dict(slot_mapping)

        batch_descriptor = BatchDescriptor(num_tokens=total_tokens)
        with set_forward_context(
            attn_metadata=attn_metadata,
            vllm_config=self._draft_vllm_config,
            num_tokens=total_tokens,
            slot_mapping=slot_mapping_dict,
            batch_descriptor=batch_descriptor,
        ):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=all_positions,
            )

        # Compute logits
        if hasattr(self.model, "compute_logits"):
            all_logits = self.model.compute_logits(hidden_states)
        elif hasattr(self.model, "lm_head"):
            all_logits = self.model.lm_head(hidden_states)
        else:
            all_logits = torch.matmul(
                hidden_states,
                self.model.get_input_embeddings().weight.T,
            )
        all_logits = all_logits[:, :V]

        # Reshape: [B*(K+1), V] → [B, K+1, V]
        # Take positions 1..K (the mask positions) as draft logits
        all_logits_2d = all_logits.view(B, Kp1, V)
        draft_logits = all_logits_2d[:, 1:, :]  # [B, K, V]

        # Apply Saguaro rescaling if configured
        if saguaro_sampler is not None:
            for step in range(K):
                draft_logits[:, step] = saguaro_sampler.apply(
                    draft_logits[:, step], temperature=temperature,
                )

        # Sample draft tokens (greedy)
        draft_tokens = draft_logits.argmax(dim=-1)  # [B, K]

        # Update tracked sequence lengths
        for i, sid in enumerate(seq_ids.tolist()):
            self._seq_lens[int(sid)] = int(positions[i].item()) + Kp1

        return draft_tokens, draft_logits
