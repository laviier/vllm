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

import os
import time
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger

logger = init_logger(__name__)

_DISAGG_DEBUG = os.environ.get("DISAGG_EAGLE_DEBUG", "0") == "1"


class DraftModelRunner:
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
        self.method = spec_config.method
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
        # Dedicated blocks reserved by the last _build_next_cache call.
        # Recycled at the start of the next _build_next_cache.
        self._dedicated_blocks: list[int] = []
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

        # CUDA graph state for eagle_forward (with hidden_states input)
        self._eagle_graphs: dict[int, dict] = {}  # bs → graph + buffers
        self._eagle_graphs_captured = False

    # ---------------------------------------------------------------
    # KV cache snapshot / rollback for tree decode
    # ---------------------------------------------------------------

    def save_kv_snapshot(self, seq_ids: list[int]) -> None:
        """Save KV cache state for the given sequences.

        Saves the current sequence lengths so we can roll back after
        branching in tree decode. The KV cache data itself doesn't
        need copying — we reuse the same physical blocks and just
        overwrite positions beyond the snapshot point on each branch.

        Args:
            seq_ids: Sequence IDs to snapshot.
        """
        self._kv_snapshot = {
            sid: self._seq_lens.get(sid, 0) for sid in seq_ids
        }

    def rollback_kv(self, seq_ids: list[int]) -> None:
        """Roll back KV cache state to the last snapshot.

        Restores sequence lengths to their snapshot values. The actual
        KV entries beyond the snapshot point are stale but harmless —
        they'll be overwritten by the next branch's decode steps since
        slot_mapping directs writes to the correct positions.

        Args:
            seq_ids: Sequence IDs to roll back.
        """
        if self._kv_snapshot is None:
            return
        for sid in seq_ids:
            if sid in self._kv_snapshot:
                self._seq_lens[sid] = self._kv_snapshot[sid]

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
        B = tokens.shape[0]
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

    def eagle_tree_decode_step(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_ids_expanded: torch.Tensor,
        block_tables: torch.Tensor,
        hidden_states: torch.Tensor,
        max_seq_len_hint: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one EAGLE tree decode step with hidden state feedback.

        Like ``tree_decode_step`` but passes hidden states to the EAGLE
        head and returns both logits and output hidden states for the
        next depth level.  Does NOT use CUDA graphs (EAGLE forward
        requires hidden_states input which graphs don't capture).

        Args:
            input_ids: [N] — input token IDs for each branch.
            positions: [N] — position of each token.
            seq_lens: [N] — context length for each branch.
            seq_ids_expanded: [N] — sequence IDs (one per branch).
            block_tables: [N, M] — per-branch block tables.
            hidden_states: [N, hidden_size] — hidden states from the
                target model (depth 0) or previous EAGLE step.
            max_seq_len_hint: Optional hint for max sequence length.

        Returns:
            Tuple of (logits, out_hidden_states):
              - logits: [N, V] — logits for each branch.
              - out_hidden_states: [N, hidden_size] — hidden states to
                feed into the next depth level.
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

        # Eager path only — CUDA graphs don't support hidden_states input
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
            output = self.model(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
            )

        # MTP returns a single tensor; EAGLE/EAGLE3 returns a tuple.
        if self.method == "mtp":
            last_hs = output
            out_hs = output
        else:
            last_hs, out_hs = output

        # Compute logits from last_hidden_states
        if hasattr(self.model, "compute_logits"):
            logits = self.model.compute_logits(last_hs)
        elif hasattr(self.model, "lm_head"):
            logits = self.model.lm_head(last_hs)
        else:
            logits = torch.matmul(
                last_hs,
                self.model.get_input_embeddings().weight.T,
            )

        return logits[:, :self.vocab_size], out_hs

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

        # In the standard EAGLE flow, embed_tokens and lm_head are
        # shared from the target model via load_eagle_model().  In
        # disagg mode, the EAGLE model is loaded independently on a
        # separate GPU, so we must load the target's embed_tokens and
        # lm_head weights from the target model files to match the
        # co-located behavior exactly.
        try:
            self._load_target_embed_and_lm_head()
        except Exception as e:
            logger.warning(
                "Failed to load target embed/lm_head: %s", e,
                exc_info=True,
            )

        # Store the draft vllm_config — attention layers register
        # themselves in compilation_config.static_forward_context
        # during model construction, so we must use this config
        # (not the original) for set_forward_context() calls.
        self._draft_vllm_config = draft_vllm_config

        dt = time.perf_counter() - t0
        logger.info("Draft model loaded in %.1f seconds.", dt)

        # Verify embedding weights were loaded (not random).
        # In the standard EAGLE flow, embed_tokens is shared from the
        # target model. In disagg, we load from the model files.
        # If the model files don't include embed_tokens, the weights
        # would be uninitialized.
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            embed_w = self.model.model.embed_tokens.weight
            logger.info(
                "Draft model embed_tokens: shape=%s, norm=%.4f, "
                "mean=%.6f, std=%.6f",
                embed_w.shape,
                embed_w.float().norm().item(),
                embed_w.float().mean().item(),
                embed_w.float().std().item(),
            )
        if hasattr(self.model, 'lm_head'):
            lm_w = self.model.lm_head.weight
            logger.info(
                "Draft model lm_head: shape=%s, norm=%.4f",
                lm_w.shape, lm_w.float().norm().item(),
            )
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'fc'):
            fc_w = self.model.model.fc.weight
            logger.info(
                "Draft model fc: shape=%s, norm=%.4f",
                fc_w.shape, fc_w.float().norm().item(),
            )

        self._allocate_kv_cache()
        self._bind_kv_cache_to_attention_layers()
        self._model_loaded = True

        # Re-enable CUDA graphs with max_seq_len set to max_model_len.
        # FlashAttention uses seqused_k (tensor, updatable) for actual
        # attention bounds, and max_seqlen_k (int, baked in) as an upper
        # bound hint. Setting it to max_model_len is safe.
        try:
            self._capture_decode_graphs()
        except Exception as e:
            logger.warning(
                "CUDA graph capture failed: %s. Using eager decode.", e
            )
            self._decode_graphs_captured = False

        # Tree decode graphs also need the same fix.
        try:
            self._capture_tree_decode_graphs()
        except Exception as e:
            logger.warning(
                "Tree decode CUDA graph capture failed: %s. Using eager.", e
            )
            self._tree_decode_captured = False

        # EAGLE forward graphs (with hidden_states input)
        if self.method in ("eagle", "eagle3", "mtp"):
            try:
                self._capture_eagle_graphs()
            except Exception as e:
                logger.warning(
                    "EAGLE CUDA graph capture failed: %s. Using eager.", e
                )
                self._eagle_graphs_captured = False

    def _load_target_embed_and_lm_head(self) -> None:
        """Load target model's embed_tokens and lm_head onto the draft GPU.

        In the co-located EAGLE flow, ``load_eagle_model()`` replaces
        the EAGLE model's ``embed_tokens`` and ``lm_head`` with the
        target model's versions (shared references).  In disagg mode
        the EAGLE model lives on a separate GPU, so we cannot share
        references.  Instead we load the weights from the target
        model's safetensors files and copy them into the EAGLE model's
        parameters.

        This is critical for acceptance rate: the EAGLE head must use
        the exact same embedding and output projection as the target
        model.  Even small precision differences (e.g. the EAGLE
        model files storing float16 while the target uses bfloat16)
        can halve the acceptance rate.
        """
        import glob
        import os

        target_model_path = self.vllm_config.model_config.model
        logger.info(
            "Loading target embed_tokens/lm_head from: %s",
            target_model_path,
        )
        if not os.path.isdir(target_model_path):
            logger.warning(
                "Target model path %s is not a directory, "
                "skipping target embed/lm_head load.",
                target_model_path,
            )
            return

        try:
            from safetensors import safe_open
        except ImportError:
            logger.warning("safetensors not available, "
                           "skipping target embed/lm_head load.")
            return

        st_files = sorted(glob.glob(
            os.path.join(target_model_path, "*.safetensors")))
        if not st_files:
            logger.warning("No safetensors files in %s", target_model_path)
            return

        logger.info("Found %d safetensors files in target model dir.",
                     len(st_files))

        # Collect all target weight tensors we need
        need_embed = (hasattr(self.model, 'model')
                      and hasattr(self.model.model, 'embed_tokens'))
        need_lm_head = hasattr(self.model, 'lm_head')
        logger.info("Need embed_tokens=%s, need lm_head=%s",
                     need_embed, need_lm_head)

        loaded_embed = False
        loaded_lm_head = False

        for st_file in st_files:
            if loaded_embed and loaded_lm_head:
                break
            try:
                with safe_open(st_file, framework="pt",
                               device="cpu") as f:
                    keys = list(f.keys())
                    # Log keys from first file for debugging
                    if st_file == st_files[0]:
                        embed_keys = [k for k in keys
                                      if "embed" in k.lower()]
                        lm_keys = [k for k in keys
                                   if "lm_head" in k.lower()]
                        logger.info(
                            "First shard %s: %d keys, "
                            "embed-related=%s, lm_head-related=%s",
                            os.path.basename(st_file), len(keys),
                            embed_keys[:5], lm_keys[:5],
                        )

                    for key in keys:
                        # --- embed_tokens ---
                        if (not loaded_embed and need_embed
                                and "embed_tokens" in key
                                and "weight" in key):
                            tensor = f.get_tensor(key)
                            dst = self.model.model.embed_tokens.weight
                            nr = min(dst.shape[0], tensor.shape[0])
                            nc = min(dst.shape[1], tensor.shape[1])
                            dst.data[:nr, :nc] = (
                                tensor[:nr, :nc]
                                .to(dst.dtype).to(dst.device)
                            )
                            loaded_embed = True
                            logger.info(
                                "Loaded target embed_tokens from %s/%s "
                                "(%s → %s): rows=%d cols=%d",
                                os.path.basename(st_file), key,
                                tensor.shape, dst.shape, nr, nc,
                            )

                        # --- lm_head ---
                        if (not loaded_lm_head and need_lm_head
                                and "lm_head" in key
                                and "weight" in key):
                            tensor = f.get_tensor(key)
                            dst = self.model.lm_head.weight
                            nr = min(dst.shape[0], tensor.shape[0])
                            nc = min(dst.shape[1], tensor.shape[1])
                            dst.data[:nr, :nc] = (
                                tensor[:nr, :nc]
                                .to(dst.dtype).to(dst.device)
                            )
                            loaded_lm_head = True
                            logger.info(
                                "Loaded target lm_head from %s/%s "
                                "(%s → %s): rows=%d cols=%d",
                                os.path.basename(st_file), key,
                                tensor.shape, dst.shape, nr, nc,
                            )
            except Exception as e:
                logger.warning("Error reading %s: %s",
                               os.path.basename(st_file), e)

        if need_embed and not loaded_embed:
            logger.warning("Could not find embed_tokens in target files.")
        if need_lm_head and not loaded_lm_head:
            # lm_head might be tied to embed_tokens (weight tying).
            # Try copying embed_tokens weights to lm_head.
            if loaded_embed and need_embed:
                src = self.model.model.embed_tokens.weight
                dst = self.model.lm_head.weight
                if src.shape == dst.shape:
                    dst.data.copy_(src.data)
                    loaded_lm_head = True
                    logger.info(
                        "lm_head not found in target files; "
                        "copied from embed_tokens (weight tying)."
                    )
                else:
                    logger.warning(
                        "lm_head not found and shapes don't match "
                        "for weight tying: embed=%s, lm_head=%s",
                        src.shape, dst.shape,
                    )
            else:
                logger.warning(
                    "Could not find lm_head in target files."
                )

        logger.info(
            "Target weight load complete: embed=%s, lm_head=%s",
            loaded_embed, loaded_lm_head,
        )

    def _capture_decode_graphs(self) -> None:
        """Capture CUDA graphs for decode_step at common batch sizes.

        With hybrid swap+JIT, decode graphs are used for JIT on cache
        misses (B_miss) which is typically much smaller than B.
        """
        max_bs = min(self.max_num_seqs, 128)
        sizes = [bs for bs in [1, 2, 4, 8, 16, 32, 48, 64, 96, 128]
                 if bs <= max_bs]
        if max_bs not in sizes:
            sizes.append(max_bs)
        logger.info("Capturing CUDA graphs for decode_step: bs=%s", sizes)
        self._capture_graphs_for_sizes(sizes, self._decode_graphs)
        self._decode_graphs_captured = True
        logger.info("CUDA graphs captured for %d decode sizes.", len(sizes))

    def _capture_tree_decode_graphs(self) -> None:
        """Capture CUDA graphs for tree_decode_step at common N values.

        Tree decode processes N = B × (K+1) × F branch tokens per step.
        Includes larger sizes for high-concurrency scenarios with
        adaptive fan-out (F=1 at high B).
        """
        sizes = [18, 36, 54, 72, 90, 108, 144, 192, 256, 336, 504]
        logger.info("Capturing CUDA graphs for tree_decode_step: N=%s", sizes)
        self._capture_graphs_for_sizes(sizes, self._tree_graphs)
        self._tree_decode_captured = True
        logger.info("CUDA graphs captured for %d tree sizes.", len(sizes))

    def _capture_eagle_graphs(self) -> None:
        """Capture CUDA graphs for eagle_forward (with hidden_states).

        Unlike decode_step graphs, these include hidden_states as an
        additional input buffer. Captured for common batch sizes.
        """
        max_bs = min(self.max_num_seqs, 8)
        sizes = [bs for bs in [1, 2, 4, 8] if bs <= max_bs]
        logger.info("Capturing CUDA graphs for eagle_forward: bs=%s", sizes)

        max_n = max(sizes)
        g_input_ids = torch.zeros(max_n, dtype=torch.int64, device=self.device)
        g_positions = torch.zeros(max_n, dtype=torch.long, device=self.device)
        g_slot_mapping = torch.zeros(max_n, dtype=torch.int64, device=self.device)
        g_seq_lens = torch.ones(max_n, dtype=torch.int32, device=self.device)
        g_block_tables = torch.zeros(
            max_n, self.max_num_blocks, dtype=torch.int32, device=self.device)
        g_hidden_states = torch.zeros(
            max_n, self.hidden_size, dtype=self.dtype, device=self.device)
        g_query_start_loc = torch.arange(
            max_n + 1, dtype=torch.int32, device=self.device)
        # Output buffers
        g_out_last_hs = torch.zeros(
            max_n, self.hidden_size, dtype=self.dtype, device=self.device)
        g_out_prenorm = torch.zeros(
            max_n, self.hidden_size, dtype=self.dtype, device=self.device)

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
            slot_mapping_dict = self._build_slot_mapping_dict(g_slot_mapping[:n])
            batch_descriptor = BatchDescriptor(num_tokens=n)

            # Warmup
            with set_forward_context(
                attn_metadata=attn_metadata,
                vllm_config=self._draft_vllm_config,
                num_tokens=n,
                slot_mapping=slot_mapping_dict,
                batch_descriptor=batch_descriptor,
            ):
                output = self.model(
                    input_ids=g_input_ids[:n],
                    positions=g_positions[:n],
                    hidden_states=g_hidden_states[:n],
                )
            if self.method != "mtp":
                g_out_last_hs[:n], g_out_prenorm[:n] = output
            else:
                g_out_last_hs[:n] = output
                g_out_prenorm[:n] = output

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
                    output = self.model(
                        input_ids=g_input_ids[:n],
                        positions=g_positions[:n],
                        hidden_states=g_hidden_states[:n],
                    )
                if self.method != "mtp":
                    g_out_last_hs[:n], g_out_prenorm[:n] = output
                else:
                    g_out_last_hs[:n] = output
                    g_out_prenorm[:n] = output

            if self._decode_graph_pool is None:
                self._decode_graph_pool = graph.pool()

            self._eagle_graphs[n] = {
                "graph": graph,
                "input_ids": g_input_ids,
                "positions": g_positions,
                "slot_mapping": g_slot_mapping,
                "seq_lens": g_seq_lens,
                "block_tables": g_block_tables,
                "hidden_states": g_hidden_states,
                "query_start_loc": g_query_start_loc,
                "out_last_hs": g_out_last_hs,
                "out_prenorm": g_out_prenorm,
                "attn_metadata": attn_metadata,
                "slot_mapping_dict": slot_mapping_dict,
                "batch_descriptor": batch_descriptor,
            }
            torch.cuda.synchronize()

        self._eagle_graphs_captured = True
        logger.info("CUDA graphs captured for %d eagle sizes.", len(sizes))

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

    def _allocate_kv_cache(self) -> None:
        """Allocate KV cache on the draft GPU."""
        torch.cuda.empty_cache()

        free_mem, _ = torch.cuda.mem_get_info(self.device)
        usable_bytes = int(free_mem * 0.8)

        bytes_per_block = (
            2 * self.num_layers * self.block_size
            * self.num_kv_heads * self.head_dim
            * torch.finfo(self.dtype).bits // 8
        )

        self.num_kv_blocks = max(1, usable_bytes // bytes_per_block)

        logger.info(
            "Draft KV cache: %d blocks × %d tokens/block = %d max tokens, "
            "%.1f GB",
            self.num_kv_blocks, self.block_size,
            self.num_kv_blocks * self.block_size,
            self.num_kv_blocks * bytes_per_block / 1e9,
        )

        # Allocate KV cache as list of [2, num_blocks, block_size, num_kv_heads, head_dim]
        # Following vLLM's KV cache layout for paged attention
        self.kv_caches = []
        for _ in range(self.num_layers):
            kv = torch.zeros(
                2,  # K and V
                self.num_kv_blocks,
                self.block_size,
                self.num_kv_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            self.kv_caches.append(kv)

    def _bind_kv_cache_to_attention_layers(self) -> None:
        """Bind allocated KV cache tensors to the model's attention layers.

        In V1, each Attention layer looks up its KV cache from
        `layer.kv_cache[virtual_engine]` during the forward pass.
        After model construction, each layer registers itself in
        `compilation_config.static_forward_context[layer_name]`.
        We iterate over those registered layers and assign each one
        a KV cache tensor from our allocated pool.

        The KV cache tensors are assigned in order of layer index.
        """
        forward_ctx = (
            self._draft_vllm_config.compilation_config.static_forward_context
        )
        if not forward_ctx:
            logger.warning(
                "No attention layers found in static_forward_context. "
                "KV cache will not be bound."
            )
            return

        from vllm.model_executor.layers.attention import Attention
        from vllm.model_executor.models.utils import extract_layer_index

        # Sort layers by layer index to match our kv_caches list order
        attn_layers = []
        for layer_name, layer in forward_ctx.items():
            if isinstance(layer, Attention):
                try:
                    idx = extract_layer_index(layer_name)
                except (ValueError, IndexError):
                    idx = len(attn_layers)
                attn_layers.append((idx, layer_name, layer))

        attn_layers.sort(key=lambda x: x[0])

        assert self.kv_caches is not None
        if len(attn_layers) != len(self.kv_caches):
            logger.warning(
                "Mismatch: %d attention layers vs %d KV cache tensors. "
                "Binding min(%d, %d) layers.",
                len(attn_layers), len(self.kv_caches),
                len(attn_layers), len(self.kv_caches),
            )

        num_bind = min(len(attn_layers), len(self.kv_caches))
        for i in range(num_bind):
            _, layer_name, layer = attn_layers[i]
            # In V1/V2, Attention.kv_cache is a single tensor (not a list).
            layer.kv_cache = self.kv_caches[i]

        logger.info(
            "Bound KV cache to %d attention layers.", num_bind
        )

    # ---------------------------------------------------------------
    # Block management
    # ---------------------------------------------------------------

    def _alloc_one_block(self) -> int:
        """Return a single free block ID, preferring the free list."""
        if self._free_list:
            return self._free_list.pop()
        if self._next_free_block >= self.num_kv_blocks:
            raise RuntimeError(
                f"Draft KV cache exhausted: {self._next_free_block} >= "
                f"{self.num_kv_blocks} (free_list empty)"
            )
        blk = self._next_free_block
        self._next_free_block += 1
        return blk

    def allocate_blocks(self, seq_id: int, num_tokens: int) -> list[int]:
        """Allocate KV cache blocks for a sequence.

        If the seq_id already has blocks (e.g. a new request reusing
        the same seq_id after the previous request finished), the old
        blocks are recycled onto the free list first.

        Raises:
            ValueError: If seq_id exceeds GPU block table capacity.
            RuntimeError: If KV cache blocks are exhausted.
        """
        if seq_id >= self._block_table_gpu.shape[0]:
            raise ValueError(
                f"seq_id {seq_id} exceeds GPU block table capacity "
                f"({self._block_table_gpu.shape[0]}). Increase max_num_seqs."
            )

        # Recycle old blocks for this seq_id if any
        if seq_id in self._block_tables:
            self._free_list.extend(self._block_tables.pop(seq_id))

        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        blocks = [self._alloc_one_block() for _ in range(num_blocks_needed)]
        self._block_tables[seq_id] = blocks
        # Update GPU-resident block table (zero stale entries first)
        self._block_table_gpu[seq_id].zero_()
        self._block_table_gpu[seq_id, :len(blocks)] = torch.tensor(
            blocks, dtype=torch.int32, device=self.device
        )
        return blocks

    def free_blocks(self, seq_id: int) -> None:
        """Free KV cache blocks for a completed sequence."""
        old_blocks = self._block_tables.pop(seq_id, None)
        if old_blocks:
            self._free_list.extend(old_blocks)
        self._seq_lens.pop(seq_id, None)
        # Zero out GPU-resident block table row
        if seq_id < self._block_table_gpu.shape[0]:
            self._block_table_gpu[seq_id].zero_()

    def ensure_blocks(self, seq_id: int, num_tokens: int) -> None:
        """Grow block allocation for a sequence if needed.

        Called during decode to ensure enough blocks are allocated for
        the current sequence length. Only allocates NEW blocks beyond
        what's already allocated — does NOT free or reallocate existing
        blocks.

        Args:
            seq_id: Sequence ID.
            num_tokens: Total number of tokens the sequence needs
                (prompt + generated so far + headroom).
        """
        if seq_id not in self._block_tables:
            # No blocks allocated yet — use allocate_blocks instead
            self.allocate_blocks(seq_id, num_tokens)
            return

        current_blocks = len(self._block_tables[seq_id])
        needed_blocks = (num_tokens + self.block_size - 1) // self.block_size
        if needed_blocks <= current_blocks:
            return  # Already have enough

        # Allocate additional blocks
        extra = needed_blocks - current_blocks
        new_blocks = [self._alloc_one_block() for _ in range(extra)]
        self._block_tables[seq_id].extend(new_blocks)

        # Update GPU-resident block table with the new blocks
        start = current_blocks
        self._block_table_gpu[seq_id, start:start + extra] = torch.tensor(
            new_blocks, dtype=torch.int32, device=self.device
        )

    def swap_block_tables(
        self,
        seq_ids: torch.Tensor,
        branch_block_tables: torch.Tensor,
        prefix_lens: torch.Tensor,
        K: int,
    ) -> tuple[dict[int, list[int]], list[int]]:
        """Swap branch block table entries into the main block table.

        Only overwrites the write-range columns (logical blocks that the
        branch's tree decode wrote into). Columns before the write range
        keep their original main-sequence block IDs.

        Args:
            seq_ids: [B] — sequence IDs to swap.
            branch_block_tables: [B, M] — branch block tables from cache.
            prefix_lens: [B] — prefix length for each branch.
            K: Number of speculative tokens.

        Returns:
            Tuple of:
              owned_blocks: {seq_id: [block_ids]} — dedicated blocks now in main.
              displaced: list of block IDs that were overwritten (for recycling).
        """
        bs = self.block_size
        M = self.max_num_blocks
        B = seq_ids.shape[0]
        owned_blocks: dict[int, list[int]] = {}
        displaced: list[int] = []

        for i in range(B):
            sid = int(seq_ids[i].item())
            if sid >= self._block_table_gpu.shape[0]:
                logger.warning(
                    "swap_block_tables: seq_id %d out of bounds, skipping", sid
                )
                continue

            prefix_len = int(prefix_lens[i].item())
            first_write_blk = prefix_len // bs
            last_write_blk = (prefix_len + K - 1) // bs

            owned = []
            for blk_idx in range(first_write_blk, min(last_write_blk + 1, M)):
                # Record the block being displaced (skip 0 = uninitialized)
                if sid in self._block_tables and blk_idx < len(self._block_tables[sid]):
                    old_blk = self._block_tables[sid][blk_idx]
                    if old_blk != 0:
                        displaced.append(old_blk)

                new_block_id = int(branch_block_tables[i, blk_idx].item())
                self._block_table_gpu[sid, blk_idx] = new_block_id
                owned.append(new_block_id)

                # Update Python dict
                if sid in self._block_tables:
                    while len(self._block_tables[sid]) <= blk_idx:
                        self._block_tables[sid].append(0)
                    self._block_tables[sid][blk_idx] = new_block_id

            owned_blocks[sid] = owned

        return owned_blocks, displaced

    def release_owned_blocks(
        self, seq_id: int, owned_blocks: list[int]
    ) -> None:
        """Release dedicated blocks that were swapped into main.

        Recycles the block IDs onto the free list so they can be
        reused by future allocations.

        Args:
            seq_id: Sequence ID that owned these blocks.
            owned_blocks: Block IDs to release.
        """
        if owned_blocks:
            self._free_list.extend(owned_blocks)

    def recycle_dedicated_blocks(self) -> None:
        """Recycle dedicated tree-decode blocks from the previous round.

        Always adds dedicated blocks to the free list and attempts
        compaction. Previous approach of rewinding the bump pointer
        failed when blocks were swapped out by cache hits.
        """
        if self._dedicated_blocks:
            self._free_list.extend(self._dedicated_blocks)
            self._dedicated_blocks = []
        self._try_compact()

    def _try_compact(self) -> None:
        """Attempt to rewind the bump pointer by draining the free list.

        If the free list contains blocks at the top of the bump range,
        remove them and rewind the pointer. This recovers contiguous
        headroom for dedicated block allocation.
        """
        if not self._free_list:
            return
        free_set = set(self._free_list)
        # Rewind as far as possible from the top
        while (self._next_free_block > 0
               and (self._next_free_block - 1) in free_set):
            self._next_free_block -= 1
            free_set.discard(self._next_free_block)
        self._free_list = list(free_set)

    def reserve_dedicated_blocks(self, block_ids: list[int]) -> None:
        """Track dedicated blocks allocated for tree decode.

        These blocks are reserved until the next _build_next_cache call,
        at which point they're recycled via recycle_dedicated_blocks().
        Blocks that get swapped into main are removed from this list
        by exclude_from_dedicated().
        """
        self._dedicated_blocks = block_ids

    def exclude_from_dedicated(self, owned_blocks: list[int]) -> None:
        """Remove swapped blocks from the dedicated list.

        When blocks are swapped into main block tables, they become
        'owned' by the sequence and are tracked in SeqSwapRecord.
        Remove them from _dedicated_blocks to prevent double-free.
        """
        if owned_blocks and self._dedicated_blocks:
            owned_set = set(owned_blocks)
            self._dedicated_blocks = [
                b for b in self._dedicated_blocks if b not in owned_set
            ]

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
        seq_ids_list = seq_ids.tolist()

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
    def eagle_prefill(
        self,
        input_ids: torch.Tensor,
        num_tokens_per_seq: torch.Tensor,
        seq_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        position_offsets: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run EAGLE head prefill with prompt tokens and hidden states.

        Applies the EAGLE token-conditioning shift: token at position j
        gets conditioning from the target's hidden state at position j-1.
        So we skip the first token per sequence and drop the last hidden
        state per sequence, processing n-1 tokens per sequence.

        Args:
            input_ids: [total_tokens] — flattened input token IDs.
            num_tokens_per_seq: [B] — per-sequence token counts.
            seq_ids: [B] — sequence IDs for block allocation.
            hidden_states: [total_tokens, hidden_size] — target model
                hidden states (already projected through fc for EAGLE3).
            position_offsets: Optional per-sequence position offsets.
                When prefix caching is active, the suffix tokens start
                at position prefix_len, not 0. Each entry is the
                prefix_len for that sequence (0 for full prefill).
            input_ids: [total_tokens] — flattened input token IDs.
            num_tokens_per_seq: [B] — per-sequence token counts.
            seq_ids: [B] — sequence IDs for block allocation.
            hidden_states: [total_tokens, hidden_size] — target model
                hidden states for all prompt tokens (already projected
                through combine_hidden_states for EAGLE3).

        Returns:
            Tuple of (last_hidden_states, out_hidden_states) from the
            EAGLE model's last token per sequence.
        """
        assert self._model_loaded, "Call load_model() first"
        B = num_tokens_per_seq.shape[0]
        total_orig = input_ids.shape[0]
        seq_ids_list = seq_ids.tolist()

        # EAGLE token-conditioning shift:
        # Token at position j gets conditioning from target hidden state
        # at position j-1.  So we:
        #   - Skip the first input token per sequence
        #   - Drop the last hidden state per sequence
        # This gives us n-1 tokens per sequence.
        shifted_ids_parts = []
        shifted_hs_parts = []
        shifted_num_tokens = []
        offset = 0
        for i in range(B):
            n = int(num_tokens_per_seq[i].item())
            # Skip first token, take tokens[1:n]
            shifted_ids_parts.append(input_ids[offset + 1:offset + n])
            # Drop last hidden state, take hs[0:n-1]
            shifted_hs_parts.append(hidden_states[offset:offset + n - 1])
            shifted_num_tokens.append(n - 1)
            offset += n

        shifted_input_ids = torch.cat(shifted_ids_parts, dim=0)
        shifted_hidden_states = torch.cat(shifted_hs_parts, dim=0)
        shifted_num_tokens_t = torch.tensor(
            shifted_num_tokens, dtype=torch.int32, device=self.device)
        total = shifted_input_ids.shape[0]

        # Allocate blocks for the full prompt + headroom.
        initial_headroom = 256
        for i in range(B):
            n = int(num_tokens_per_seq[i].item())
            sid = int(seq_ids[i].item())
            pos_off = position_offsets[i] if position_offsets else 0
            full_len = n + pos_off
            self.allocate_blocks(sid, full_len + initial_headroom)

        # Build positions: [total_shifted_tokens]
        # Positions are offset..offset+n-2 for each sequence
        # (n-1 tokens after shift, starting at position_offset)
        positions = torch.zeros(total, dtype=torch.long, device=self.device)
        offset = 0
        expanded_seq_ids = []
        for i in range(B):
            n_shifted = shifted_num_tokens[i]
            pos_off = position_offsets[i] if position_offsets else 0
            positions[offset:offset + n_shifted] = torch.arange(
                n_shifted, device=self.device) + pos_off
            expanded_seq_ids.extend([int(seq_ids[i].item())] * n_shifted)
            offset += n_shifted

        # Compute slot mapping
        slot_mapping = self._compute_slot_mapping(positions, expanded_seq_ids)

        # Build block table
        block_tables = self._get_block_table_tensor(seq_ids)

        # Build FlashAttention metadata for prefill.
        # seq_lens_tensor: the total KV context each sequence can attend to.
        # With position offsets (prefix caching), this is offset + n_shifted
        # because the token at position offset+j attends to KV[0..offset+j].
        # Note: KV[0..offset-1] are uninitialized (prefix not in EAGLE cache),
        # but FlashAttention with causal masking only reads KV up to the
        # query position, and the query tokens start at offset.
        if position_offsets:
            seq_lens_with_offset = torch.tensor(
                [position_offsets[i] + shifted_num_tokens[i] for i in range(B)],
                dtype=torch.int32, device=self.device)
            max_seq_len = int(seq_lens_with_offset.max().item())
        else:
            seq_lens_with_offset = shifted_num_tokens_t
            max_seq_len = max(shifted_num_tokens)
        max_query_len = max(shifted_num_tokens)
        query_start_loc = torch.zeros(
            B + 1, dtype=torch.int32, device=self.device)
        torch.cumsum(
            shifted_num_tokens_t, dim=0, out=query_start_loc[1:])

        attn_metadata = self._build_flash_attn_metadata(
            num_tokens=total,
            seq_lens_tensor=seq_lens_with_offset,
            max_seq_len=max_seq_len,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            block_table=block_tables,
            slot_mapping=slot_mapping,
        )
        slot_mapping_dict = self._build_slot_mapping_dict(slot_mapping)

        # Run EAGLE model forward with shifted inputs
        batch_descriptor = BatchDescriptor(num_tokens=total)
        with set_forward_context(
            attn_metadata=attn_metadata,
            vllm_config=self._draft_vllm_config,
            num_tokens=total,
            slot_mapping=slot_mapping_dict,
            batch_descriptor=batch_descriptor,
        ):
            output = self.model(
                input_ids=shifted_input_ids,
                positions=positions,
                hidden_states=shifted_hidden_states,
            )

        if self.method == "mtp":
            last_hs = output
            out_hs = output
        else:
            last_hs, out_hs = output

        # Set _seq_lens to offset + n-1 (number of KV entries after
        # shifted prefill). For prefix-cached requests, this accounts
        # for the prefix offset so the next JIT starts at the correct
        # position.
        for i in range(B):
            sid = int(seq_ids[i].item())
            pos_off = position_offsets[i] if position_offsets else 0
            self._seq_lens[sid] = pos_off + shifted_num_tokens[i]

        # Extract last token per sequence
        last_indices = torch.cumsum(shifted_num_tokens_t, dim=0) - 1
        return last_hs[last_indices], out_hs[last_indices]

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
    def eagle_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        seq_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single EAGLE head forward pass.

        Runs the EAGLE/EAGLE3/MTP model with target (or previous-step)
        hidden states as input.  The model's KV cache is updated via
        the same paged-attention path used by ``decode_step``.

        Args:
            input_ids: [B] — input token IDs for this step.
            positions: [B] — position of each token in its sequence.
            hidden_states: [B, hidden_size] — hidden states from the
                target model (step 0) or from the previous EAGLE step.
            seq_ids: [B] — sequence IDs (used for KV cache slot mapping
                and sequence length tracking).

        Returns:
            Tuple of (last_hidden_states, hidden_states):
              - last_hidden_states: [B, hidden_size] — used for logit
                computation.
              - hidden_states: [B, hidden_size] — fed back as input to
                the next autoregressive step.
            For MTP method both elements are the same tensor.
        """
        assert self._model_loaded, "Call load_model() first"
        B = input_ids.shape[0]

        # --- Build attention metadata (same as decode_step) ---
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

        # Try CUDA graph replay for eagle_forward
        if self._eagle_graphs_captured and B in self._eagle_graphs:
            g = self._eagle_graphs[B]
            g["input_ids"][:B].copy_(input_ids)
            g["positions"][:B].copy_(positions)
            g["slot_mapping"][:B].copy_(slot_mapping)
            g["seq_lens"][:B].copy_(seq_lens)
            g["block_tables"][:B].copy_(block_tables)
            g["hidden_states"][:B].copy_(hidden_states)
            g["attn_metadata"].max_seq_len = max_seq_len
            for layer_name in g["slot_mapping_dict"]:
                g["slot_mapping_dict"][layer_name] = g["slot_mapping"][:B]
            g["graph"].replay()
            last_hidden_states = g["out_last_hs"][:B]
            out_hidden_states = g["out_prenorm"][:B]
        else:
            # Eager fallback
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
                output = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    hidden_states=hidden_states,
                )

            if self.method == "mtp":
                last_hidden_states = output
                out_hidden_states = output
            else:
                last_hidden_states, out_hidden_states = output

        # Note: _seq_lens is NOT updated here to avoid per-step
        # Python overhead. The caller (eagle_sequential_speculate)
        # updates _seq_lens after all K steps complete.

        return last_hidden_states, out_hidden_states

    @torch.inference_mode()
    def eagle_sequential_speculate(
        self,
        recovery_tokens: torch.Tensor,
        positions: torch.Tensor,
        seq_ids: torch.Tensor,
        num_steps: int,
        hidden_states: torch.Tensor,
        temperatures: torch.Tensor | None = None,
        seeds: torch.Tensor | None = None,
        glue_prenorm: torch.Tensor | None = None,
        glue_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run K-step autoregressive EAGLE speculation.

        Step 0 feeds the target model's hidden states into the EAGLE
        head.  Steps 1..K-1 feed the EAGLE head's own output hidden
        states back as input, forming an autoregressive chain.

        When ``glue_prenorm`` and ``glue_logits`` are provided (from
        the glue decode pass), step 0 is short-circuited: the first
        draft token is sampled from ``glue_logits`` and ``glue_prenorm``
        is used as the starting hidden state for step 1+.  This matches
        the co-located EAGLE flow where step 0 processes all query
        tokens via a full forward pass and samples from the last
        position's logits.

        Sampling strategy:
        - T > 0: Gumbel sampling with position-based seeds (correlated
          with the target model's Gumbel noise for higher acceptance).
        - T == 0: Greedy argmax.

        Args:
            recovery_tokens: [B] — first input tokens (bonus tokens
                from verification).
            positions: [B] — starting positions in each sequence.
            seq_ids: [B] — sequence IDs.
            num_steps: K — number of draft tokens to generate.
            hidden_states: [B, hidden_size] — target model hidden
                states for step 0.
            temperatures: [B] — per-request sampling temperatures.
                ``None`` means greedy for all requests.
            seeds: [B] — per-request random seeds for Gumbel noise.
                Required when any temperature > 0.
            glue_prenorm: [B, hidden_size] — EAGLE model's prenorm
                output from glue decode at the recovery position.
                When provided, step 0 is skipped and this is used
                as the starting hidden state for step 1.
            glue_logits: [B, V] — logits from glue decode at the
                recovery position. When provided, step 0 token is
                sampled from these instead of running the EAGLE model.

        Returns:
            draft_tokens: [B, K] — generated draft token IDs.
            draft_logits: [B, K, V] — logits at each step.
            draft_prenorms: [B, K, hidden_size] — per-step prenorm
                outputs from the EAGLE head (self-conditioned hidden
                states). Used by ``_build_next_cache`` for tree decode.
                ``None`` for MTP method.
        """
        B = recovery_tokens.shape[0]
        V = self.vocab_size

        draft_tokens = torch.zeros(
            B, num_steps, dtype=torch.int64, device=self.device,
        )
        draft_logits = torch.zeros(
            B, num_steps, V, dtype=self.dtype, device=self.device,
        )

        # Track per-step prenorms for tree decode cache building.
        hs_dim = hidden_states.shape[-1] if self.method != "mtp" else 0
        draft_prenorms: torch.Tensor | None = None
        if self.method != "mtp" and hs_dim > 0:
            draft_prenorms = torch.zeros(
                B, num_steps, hs_dim,
                dtype=self.dtype, device=self.device,
            )

        current_ids = recovery_tokens
        current_pos = positions.clone()
        current_hs = hidden_states  # target hidden states for step 0

        # Determine if we need Gumbel sampling (any temperature > 0).
        use_gumbel = (
            temperatures is not None
            and seeds is not None
            and (temperatures > 0).any()
        )

        if use_gumbel:
            from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
            idx_mapping = torch.arange(
                B, dtype=torch.int64, device=self.device,
            )

        # When glue decode results are available, use them for step 0.
        # The glue decode already ran the EAGLE model forward pass over
        # extend + recovery tokens, producing prenorm and logits at the
        # recovery position.  Using these directly matches the co-located
        # flow where step 0 processes all query tokens via run_model and
        # samples from the last position's logits.
        use_glue_for_step0 = (
            glue_prenorm is not None
            and glue_logits is not None
        )
        start_step = 0

        if use_glue_for_step0:
            # Sample step 0 token from glue decode logits.
            # We still run eagle_forward below (step 0 is NOT skipped
            # from the loop) to populate the KV cache at this position.
            # But we use glue logits for sampling and glue prenorm as
            # the starting hidden state for step 1+.
            pos0 = positions
            if use_gumbel:
                step0_tokens = gumbel_sample(
                    logits=glue_logits,
                    expanded_idx_mapping=idx_mapping,
                    temperature=temperatures,
                    seed=seeds,
                    pos=pos0,
                    apply_temperature=True,
                )
            else:
                step0_tokens = glue_logits.argmax(dim=-1)

            if _DISAGG_DEBUG:
                top5_vals, top5_ids = torch.topk(glue_logits[0], 5)
                logger.info(
                    "[DISAGG_DIAG][CP7] step=0 (glue) "
                    "input_hs_norm=%.6f "
                    "output_prenorm_norm=%.6f "
                    "top5_ids=%s top5_vals=%s "
                    "sampled_token=%d pos=%d",
                    hidden_states[0].float().norm().item(),
                    glue_prenorm[0].float().norm().item(),
                    top5_ids.tolist(), top5_vals.tolist(),
                    step0_tokens[0].item(), pos0[0].item(),
                )

        # Pre-compute slot mappings for all K positions to avoid
        # per-step Python overhead.
        seq_ids_long = seq_ids.to(torch.int64)
        all_positions = [positions + step for step in range(num_steps)]

        for step in range(start_step, num_steps):
            pos = all_positions[step]

            if _DISAGG_DEBUG:
                logger.info(
                    "[DISAGG_DIAG][CP7] step=%d input_hs_norm=%.6f",
                    step, current_hs.float().norm().item(),
                )

            # Compute slot mapping (vectorized, no Python loop)
            logical_blocks = (pos // self.block_size).to(torch.int64)
            offsets = (pos % self.block_size).to(torch.int64)
            physical_blocks = self._block_table_gpu[
                seq_ids_long, logical_blocks
            ].to(torch.int64)
            slot_mapping = physical_blocks * self.block_size + offsets
            block_tables = self._block_table_gpu[seq_ids_long]
            seq_lens = (pos + 1).to(torch.int32)
            max_seq_len = int(seq_lens.max().item())

            # Try CUDA graph replay
            if self._eagle_graphs_captured and B in self._eagle_graphs:
                g = self._eagle_graphs[B]
                g["input_ids"][:B].copy_(current_ids)
                g["positions"][:B].copy_(pos)
                g["slot_mapping"][:B].copy_(slot_mapping)
                g["seq_lens"][:B].copy_(seq_lens)
                g["block_tables"][:B].copy_(block_tables)
                g["hidden_states"][:B].copy_(current_hs)
                g["attn_metadata"].max_seq_len = max_seq_len
                for ln in g["slot_mapping_dict"]:
                    g["slot_mapping_dict"][ln] = g["slot_mapping"][:B]
                g["graph"].replay()
                last_hs = g["out_last_hs"][:B]
                current_hs = g["out_prenorm"][:B]
            else:
                last_hs, current_hs = self.eagle_forward(
                    current_ids, pos, current_hs, seq_ids,
                )

            # Store per-step prenorm (self-conditioned hidden state)
            if draft_prenorms is not None:
                draft_prenorms[:, step] = current_hs

            # When using glue shortcut for step 0, override the
            # eagle_forward outputs with glue decode results.
            # eagle_forward still ran above to populate the KV cache
            # at this position, but we use glue logits for sampling
            # and glue prenorm as the hidden state for step 1+.
            if use_glue_for_step0 and step == 0:
                current_hs = glue_prenorm
                if draft_prenorms is not None:
                    draft_prenorms[:, 0] = glue_prenorm
                draft_logits[:, 0] = glue_logits
                draft_tokens[:, 0] = step0_tokens
                current_ids = step0_tokens
                continue

            # Compute logits and sample
            if hasattr(self.model, "compute_logits"):
                logits = self.model.compute_logits(last_hs)
            elif hasattr(self.model, "lm_head"):
                logits = self.model.lm_head(last_hs)
            else:
                logits = torch.matmul(
                    last_hs,
                    self.model.get_input_embeddings().weight.T,
                )
            logits = logits[:, :V]
            draft_logits[:, step] = logits

            if use_gumbel:
                next_tokens = gumbel_sample(
                    logits=logits,
                    expanded_idx_mapping=idx_mapping,
                    temperature=temperatures,
                    seed=seeds,
                    pos=pos,
                    apply_temperature=True,
                )
            else:
                next_tokens = logits.argmax(dim=-1)

            draft_tokens[:, step] = next_tokens
            current_ids = next_tokens

            if _DISAGG_DEBUG:
                top5_vals, top5_ids = torch.topk(logits[0], 5)
                logger.info(
                    "[DISAGG_DIAG][CP7] step=%d output_prenorm_norm=%.6f "
                    "top5_ids=%s top5_vals=%s "
                    "sampled_token=%d pos=%d",
                    step, current_hs.float().norm().item(),
                    top5_ids.tolist(), top5_vals.tolist(),
                    next_tokens[0].item(), pos[0].item(),
                )

        # Update _seq_lens after all K steps (deferred from eagle_forward)
        final_pos = all_positions[-1]
        for i, sid in enumerate(seq_ids.tolist()):
            self._seq_lens[int(sid)] = int(final_pos[i].item()) + 1

        return draft_tokens, draft_logits, draft_prenorms

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
        to predict tokens at <mask> positions (P-EAGLE style).

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
