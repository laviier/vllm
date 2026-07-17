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
from vllm.v1.worker.gpu.spec_decode.disagg_draft.attn_metadata import (
    DraftAttnMetadataMixin,
)
from vllm.v1.worker.gpu.spec_decode.disagg_draft.cuda_graphs import (
    DraftCudaGraphMixin,
)
from vllm.v1.worker.gpu.spec_decode.disagg_draft.kv_cache_manager import (
    DraftKVCacheMixin,
)

logger = init_logger(__name__)


class DraftModelRunner(
    DraftKVCacheMixin,
    DraftAttnMetadataMixin,
    DraftCudaGraphMixin,
):
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
        # Total geometric fan-out budget per sequence (matches the
        # OutcomePredictor's total_budget on the server side). Used by
        # the CUDA graph capture mixin to size graphs for parallel
        # fanout (N×K) and KV cleanup (N×(K-1)) call shapes.
        self._sum_fan_out = (
            spec_config.disagg_fan_out * (self._num_spec_tokens + 1)
        )

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

        # ---- KV-copy graph state (branch-block KV copy in cache_build) ---
        # Fancy-index copy across all attention layers costs ~4 ms/round
        # under 3V+1D c=8 K=4 because each layer fires its own kernel
        # (28 layers × ~150 μs launch overhead). Batching into one CUDA
        # graph replay cuts that to ~100 μs. Buffers are sized to the
        # worst-case ``MAX_BRANCHES × max_blocks_per_branch``; each new
        # ``n_copies`` value captures its own graph on first use.
        self._kv_copy_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._kv_copy_src_buf: torch.Tensor | None = None
        self._kv_copy_dst_buf: torch.Tensor | None = None
        self._kv_copy_graph_pool = None

    # Upper bound on n for KV copy: max branches × max blocks/branch.
    # MAX_BRANCHES = 504 (DraftServer.MAX_BRANCHES). blocks_per_branch
    # = (K + block_size) // block_size + 1 — 3 covers K up to ~2×bs.
    _KV_COPY_BUF_MAX = 504 * 4

    def _ensure_kv_copy_buffers(self, max_n: int) -> None:
        """One-shot allocation of persistent src/dst index buffers.

        Buffers are sized to the worst case so we never reallocate;
        that keeps every captured graph valid across all ``n`` values
        (each graph indexes into a stable underlying tensor via a
        ``[:n]`` view).
        """
        if self._kv_copy_src_buf is not None:
            return
        cap = max(max_n, self._KV_COPY_BUF_MAX)
        self._kv_copy_src_buf = torch.zeros(
            cap, dtype=torch.int64, device=self.device,
        )
        self._kv_copy_dst_buf = torch.zeros(
            cap, dtype=torch.int64, device=self.device,
        )

    def _capture_kv_copy_graph(self, n: int) -> torch.cuda.CUDAGraph:
        """Capture a CUDA graph that copies KV blocks for all layers
        using persistent ``self._kv_copy_src_buf[:n]`` /
        ``self._kv_copy_dst_buf[:n]`` as the fancy-index sources. The
        graph is safe to replay after overwriting those buffers with
        fresh indices via ``copy_``.
        """
        assert self._kv_copy_src_buf is not None
        assert self._kv_copy_dst_buf is not None
        assert self.kv_caches is not None
        src = self._kv_copy_src_buf[:n]
        dst = self._kv_copy_dst_buf[:n]
        # Warm-up outside capture to compile any needed autograd artifacts
        # and register the fancy-index kernels in the current stream.
        for layer_kv in self.kv_caches:
            layer_kv[dst] = layer_kv[src]
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._kv_copy_graph_pool):
            for layer_kv in self.kv_caches:
                layer_kv[dst] = layer_kv[src]
        if self._kv_copy_graph_pool is None:
            self._kv_copy_graph_pool = graph.pool()
        return graph

    def _capture_kv_copy_graphs(self) -> None:
        """Capture KV-copy graphs at expected ``n`` values.

        Under 3V+1D c=8 K=4 fan_out=3 the runtime ``n`` values are
        ``B_total × entries_per_seq × blocks_per_branch`` — a small
        set determined by config. We over-capture to cover the
        realistic B_total range (1..24) at typical
        ``blocks_per_branch`` values so cache_build always hits a
        pre-captured graph and never has to pay first-call capture
        cost on the hot path.
        """
        if self.kv_caches is None:
            return
        # Pre-allocate the biggest buffer we'll ever need.
        self._ensure_kv_copy_buffers(self._KV_COPY_BUF_MAX)
        # Enumerate plausible ``n`` = B × entries_per_seq × blocks/branch.
        # entries_per_seq = disagg_fan_out × (K+1) = sum_fan_out;
        # blocks_per_branch = (K + bs) // bs + 1 (typically 2 for K≤bs,
        # 3 for bs < K ≤ 2*bs). Capture both to be safe.
        K = self._num_spec_tokens
        bs = self.block_size
        sum_fan = self._sum_fan_out
        candidates: set[int] = set()
        for bpb in (2, 3):
            for b_total in (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32):
                n = b_total * sum_fan * bpb
                if 0 < n <= self._KV_COPY_BUF_MAX:
                    candidates.add(n)
        for n in sorted(candidates):
            try:
                self._kv_copy_graphs[n] = self._capture_kv_copy_graph(n)
            except Exception as e:
                logger.warning(
                    "KV copy graph capture for n=%d failed: %s", n, e,
                )
        logger.info(
            "KV copy graphs captured: %d sizes.",
            len(self._kv_copy_graphs),
        )

    def run_kv_copy(
        self, src_indices: torch.Tensor, dst_indices: torch.Tensor,
    ) -> None:
        """Batched fancy-index KV copy across all attention layers.

        Replays a captured CUDA graph so all ``num_layers`` copies fire
        as a single dispatch instead of ``num_layers`` separate kernel
        launches. Falls back to a per-layer eager loop if the graph
        for this size wasn't captured (e.g. capture failed at init).

        ``src_indices`` and ``dst_indices`` must be int64 tensors of
        the same length on the runner's device. Callers can (and
        should) include self-copies (``src == dst``) so shape stays
        fixed — that lets a single captured graph handle any
        hit-count pattern.
        """
        n = src_indices.numel()
        assert n == dst_indices.numel()
        if n == 0:
            return
        graph = self._kv_copy_graphs.get(n)
        if graph is None or self._kv_copy_src_buf is None:
            # Eager fallback: no captured graph for this size.
            assert self.kv_caches is not None
            for layer_kv in self.kv_caches:
                layer_kv[dst_indices] = layer_kv[src_indices]
            return
        # Copy indices into the graph-captured buffers, then replay.
        # ``non_blocking=True`` is safe because both src_indices and
        # the persistent buffer live on the same device.
        self._kv_copy_src_buf[:n].copy_(src_indices, non_blocking=True)
        assert self._kv_copy_dst_buf is not None
        self._kv_copy_dst_buf[:n].copy_(dst_indices, non_blocking=True)
        graph.replay()

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

        try:
            self._capture_kv_copy_graphs()
        except Exception as e:
            logger.warning(
                "KV copy CUDA graph capture failed: %s. Using eager.", e
            )
            # Callers fall through to the per-layer eager loop when the
            # graph dict is empty (see ``run_kv_copy``).
            self._kv_copy_graphs = {}

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
