# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Disaggregated Draft Worker for vLLM.

The draft worker runs on a separate GPU from the target model. Each
decode step, the target sends a verification outcome (k_accepted,
bonus_token) and the draft responds with K speculated tokens.

The draft worker manages its own:
- Draft model (loaded independently on a dedicated GPU)
- KV cache (separate from target, with GPU-resident block tables)
- Speculation cache (pre-computes tokens for predicted outcomes,
  built asynchronously during target verification)

Each round: cache lookup → JIT decode → send response → build cache
for next round. On cache hit, cached tokens replace JIT tokens in
the response. JIT always runs to keep the main KV cache populated.

Communication with the target uses a standalone NCCL process group.

Reference: SSD paper (arXiv:2603.03251), ssd/engine/draft_runner.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.disagg_draft.communication import (
    DisaggDraftCommand,
    DisaggDraftCommunicator,
)
from vllm.v1.worker.gpu.spec_decode.disagg_draft.draft_model_runner import DraftModelRunner
from vllm.v1.worker.gpu.spec_decode.disagg_draft.outcome_predictor import OutcomePredictor
from vllm.v1.worker.gpu.spec_decode.disagg_draft.saguaro_sampling import SaguaroSampler
from vllm.v1.worker.gpu.spec_decode.disagg_draft.speculation_cache import SpeculationCache

logger = init_logger(__name__)


@dataclass
class SeqSwapRecord:
    """Per-sequence state tracking for block table swaps.

    Tracks whether the last speculation round used a block table swap
    (cache hit) or JIT decode (cache miss). This is needed to correctly
    reset _seq_lens on the next round, since the "base length" differs
    depending on whether the previous KV came from a swap or JIT.

    Also tracks which dedicated blocks are now "owned" by the main
    sequence's block table, preventing them from being freed or reused
    until explicitly released.
    """
    last_round_was_swap: bool = False
    swap_prefix_len: int = 0
    owned_dedicated_blocks: list[int] = field(default_factory=list)


class DisaggDraftWorker:
    """Disaggregated draft worker for disagg draft speculation.

    Runs on a separate GPU and communicates with the target model
    via NCCL. Manages the draft model, speculation cache, and the
    async speculation loop.

    This class is the draft-side counterpart. It is instantiated in a
    separate process (launched by the executor) and runs an event loop
    that responds to commands from the target side.

    The target-side interface is provided by DisaggDraftTargetInterface, which
    is used by the vLLM model runner to communicate with this worker.

    Args:
        vllm_config: Full vLLM configuration.
        device: CUDA device for the draft model (e.g., cuda:4).
        communicator: NCCL communicator connected to target rank 0.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        communicator: DisaggDraftCommunicator,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.comm = communicator

        spec_config = vllm_config.speculative_config
        assert spec_config is not None

        self.K = spec_config.num_speculative_tokens
        self.vocab_size = spec_config.draft_model_config.get_vocab_size()
        self.dtype = vllm_config.model_config.dtype

        # Disagg-draft-specific config (from speculative_config extensions)
        self.fan_out = getattr(spec_config, "disagg_draft_fan_out", 3)
        self.saguaro_c = getattr(spec_config, "disagg_draft_saguaro_c", None)
        self.jit_fallback = getattr(spec_config, "disagg_draft_jit_fallback", True)

        max_batch_size = vllm_config.scheduler_config.max_num_seqs

        # Initialize components
        self.cache = SpeculationCache(
            max_batch_size=max_batch_size,
            num_speculative_tokens=self.K,
            fan_out=self.fan_out,
            vocab_size=self.vocab_size,
            device=device,
            dtype=self.dtype,
        )

        # Total fan-out budget: F * (K+1)
        total_fan_out = self.fan_out * (self.K + 1)
        self.outcome_predictor = OutcomePredictor(
            num_speculative_tokens=self.K,
            total_fan_out=total_fan_out,
            acceptance_rate=0.65,
            power_law_exponent=1.5,
            device=device,
        )
        # Override with uniform fan_out (matching reference SSD paper)
        self.outcome_predictor.fan_out_list = [self.fan_out] * (self.K + 1)
        self.outcome_predictor.max_fan_out = self.fan_out

        self.saguaro_sampler = SaguaroSampler(
            saguaro_c=self.saguaro_c,
            fan_out=self.fan_out,
            device=device,
        )

        # Draft model runner (set externally by executor)
        self.draft_model_runner: DraftModelRunner | None = None

        # Per-round state tracking
        self._step_times: list[float] = []

        # State from last speculation round, used by _build_next_cache()
        # These are set in _handle_speculation() and consumed by _build_next_cache()
        self._last_draft_tokens: torch.Tensor | None = None  # [B, K]
        self._last_draft_logits: torch.Tensor | None = None  # [B, K, V]
        self._last_bonus_tokens: torch.Tensor | None = None  # [B]

        # Base sequence lengths at the START of each speculation round
        # (before JIT or cache-hit tokens modify _seq_lens).
        # Used by _build_next_cache to compute correct rollback positions.
        self._round_base_lens: dict[int, int] = {}

        # Per-sequence swap state tracking for block table swapping.
        # Tracks whether the last round used a swap or JIT, enabling
        # correct _seq_lens reset and block ownership management.
        self._swap_states: dict[int, SeqSwapRecord] = {}

        logger.info(
            "Disagg draft Draft Worker initialized: K=%d, fan_out=%d, device=%s",
            self.K,
            self.fan_out,
            device,
        )

    def load_model(self) -> None:
        """Load the draft model onto the draft device.

        The draft model is a standalone model (e.g., Llama-3.2-1B)
        loaded independently — not sharing weights with the target.
        """
        from vllm.model_executor.model_loader import get_model

        logger.info("Loading disagg draft draft model...")
        self.model = get_model(
            vllm_config=self.vllm_config,
        )
        self.model.eval()
        logger.info("Disagg draft draft model loaded successfully.")

    @torch.inference_mode()
    def run_loop(self) -> None:
        """Main event loop for the draft worker.

        Runs continuously, processing commands from the target:
        - PREFILL: Process prefix tokens for new sequences
        - SPECULATE: Look up cache + pre-compute next round
        - FREE_SEQ: Free resources for completed sequences
        - EXIT: Shutdown

        This loop runs in its own process on the draft GPU.
        """
        logger.info("Disagg draft draft worker entering main loop.")

        while True:
            cmd = self.comm.recv_command()

            if cmd == DisaggDraftCommand.PREFILL:
                self._handle_prefill()
                continue

            elif cmd == DisaggDraftCommand.SPECULATE:
                t0 = time.perf_counter()
                self._handle_speculation()
                dt = time.perf_counter() - t0
                self._step_times.append(dt)
                continue

            elif cmd == DisaggDraftCommand.FREE_SEQ:
                self._handle_free_seq()
                continue

            elif cmd == DisaggDraftCommand.EXIT:
                self._handle_exit()
                break

            else:
                raise RuntimeError(f"Disagg draft draft worker: unknown command {cmd}")

    def _handle_prefill(self) -> None:
        """Handle PREFILL command: process prefix for new sequences."""
        input_ids, num_tokens, recv_seq_ids = self.comm.recv_prefill_data()
        B = num_tokens.shape[0]

        # Run draft model prefill if model runner is available
        if (
            self.draft_model_runner is not None
            and self.draft_model_runner._model_loaded
        ):
            seq_ids = (
                recv_seq_ids
                if recv_seq_ids is not None
                else torch.arange(B, dtype=torch.int64, device=self.device)
            )
            try:
                self.draft_model_runner.prefill(
                    input_ids=input_ids,
                    num_tokens_per_seq=num_tokens,
                    seq_ids=seq_ids,
                )
            except (RuntimeError, ValueError) as e:
                # Gracefully handle KV cache exhaustion or block table
                # overflow. The sequence will use JIT fallback.
                logger.warning("Disagg draft prefill failed: %s", e)
                return
            # Clear stale round base lengths for freshly prefilled sequences.
            for sid in seq_ids.tolist():
                self._round_base_lens.pop(int(sid), None)

    def _handle_speculation(self) -> None:
        """Handle SPECULATE command with hybrid swap+JIT strategy.

        For each sequence independently:
        - Cache hit → swap branch block tables (zero JIT latency)
        - Cache miss → run JIT decode (only on misses, not full batch)

        This hybrid approach reduces JIT batch size from B to B_miss,
        which is critical at high concurrency where B can be 50-80 but
        hit rate is ~80%, meaning JIT only runs on ~10-16 sequences.
        """
        _profile = os.environ.get("DISAGG_DRAFT_PROFILE", "0") == "1"

        # Step 1: Receive verification outcome
        B, seq_ids, k_accepted, bonus_tokens, temperatures = (
            self.comm.recv_verification_outcome()
        )

        if _profile:
            torch.cuda.synchronize(self.device)
            t_recv = time.perf_counter()

        # Step 1b: Reset _seq_lens accounting for swap state.
        if self.draft_model_runner is not None:
            runner = self.draft_model_runner
            seq_ids_list = seq_ids.tolist()
            k_accepted_list = k_accepted.tolist()

            for i, sid in enumerate(seq_ids_list):
                swap_rec = self._swap_states.get(sid)
                if swap_rec is not None and swap_rec.last_round_was_swap:
                    correct_len = (
                        swap_rec.swap_prefix_len
                        + 1
                        + int(k_accepted_list[i])
                    )
                    runner._seq_lens[sid] = correct_len
                elif sid in self._round_base_lens:
                    correct_len = (
                        self._round_base_lens[sid]
                        + 1
                        + int(k_accepted_list[i])
                    )
                    runner._seq_lens[sid] = correct_len

            # Save base lens BEFORE any JIT or swap modifies _seq_lens
            for sid in seq_ids_list:
                self._round_base_lens[sid] = runner._seq_lens.get(sid, 0)

        # Step 2: Cache lookup
        cached_tokens, cached_logits, cache_hits = self.cache.lookup(
            seq_ids=seq_ids,
            k_accepted=k_accepted,
            bonus_tokens=bonus_tokens,
        )

        # Step 3: Hybrid swap+JIT
        num_hits = int(cache_hits.sum().item())
        hit_mask = cache_hits.bool()
        miss_mask = ~hit_mask

        # Pre-allocate output tensors
        draft_tokens = torch.zeros(
            B, self.K, dtype=torch.int64, device=self.device,
        )
        draft_logits = torch.zeros(
            B, self.K, self.vocab_size, dtype=self.dtype, device=self.device,
        )

        # --- Handle cache hits: swap block tables ---
        used_swap_for_hits = False
        if num_hits > 0 and cached_logits is not None:
            hit_tables, hit_prefix_lens = self.cache.get_hit_block_tables(
                cache_hits
            )
            if hit_tables is not None and hit_prefix_lens is not None:
                # Swap only for hit sequences
                hit_seq_ids = seq_ids[hit_mask]
                owned, displaced = runner.swap_block_tables(
                    seq_ids=hit_seq_ids,
                    branch_block_tables=hit_tables,
                    prefix_lens=hit_prefix_lens,
                    K=self.K,
                )
                for blocks in owned.values():
                    runner.exclude_from_dedicated(blocks)
                if displaced:
                    runner._free_list.extend(displaced)

                # Update _seq_lens and swap state for hits
                hit_indices = hit_mask.nonzero(as_tuple=True)[0]
                for compact_i, idx in enumerate(hit_indices):
                    i = int(idx.item())
                    sid = seq_ids_list[i]
                    prefix_len = int(hit_prefix_lens[compact_i].item())
                    runner._seq_lens[sid] = prefix_len + self.K
                    self._swap_states[sid] = SeqSwapRecord(
                        last_round_was_swap=True,
                        swap_prefix_len=prefix_len,
                        owned_dedicated_blocks=owned.get(sid, []),
                    )

                # Fill hit results
                draft_tokens[hit_mask] = cached_tokens[hit_mask]
                draft_logits[hit_mask] = cached_logits[hit_mask]
                used_swap_for_hits = True

        # --- Handle cache misses: JIT only on misses ---
        B_miss = int(miss_mask.sum().item())
        if B_miss > 0:
            miss_seq_ids = seq_ids[miss_mask]
            miss_bonus = bonus_tokens[miss_mask]
            miss_temps = (temperatures[miss_mask]
                          if temperatures is not None else None)

            jit_tokens, jit_logits = self._jit_speculate(
                miss_seq_ids, miss_bonus, B_miss=B_miss,
                temperatures=miss_temps,
            )
            draft_tokens[miss_mask] = jit_tokens
            if jit_logits is not None:
                draft_logits[miss_mask] = jit_logits

            # Clear swap state for JIT sequences
            miss_indices = miss_mask.nonzero(as_tuple=True)[0]
            for idx in miss_indices:
                sid = seq_ids_list[int(idx.item())]
                self._swap_states[sid] = SeqSwapRecord(
                    last_round_was_swap=False
                )

        # If no hits were swapped, clear swap state for all
        if not used_swap_for_hits:
            for sid in seq_ids_list:
                self._swap_states[sid] = SeqSwapRecord(
                    last_round_was_swap=False
                )

        # Store for _build_next_cache (use JIT results for misses,
        # cached results for hits — JIT results are more representative
        # of the actual KV state for cache building)
        self._last_draft_tokens = draft_tokens.clone()
        self._last_draft_logits = draft_logits.clone()
        self._last_bonus_tokens = bonus_tokens.clone()

        used_swap = (B_miss == 0)  # all-swap for profiling

        # Step 4: Send speculation to target
        self.comm.send_speculation(
            cache_hits=cache_hits,
            draft_tokens=draft_tokens,
            draft_logits=draft_logits,
        )

        if _profile:
            torch.cuda.synchronize(self.device)
            t_send = time.perf_counter()

        # Step 5: Build speculation cache for NEXT round (async overlap)
        if self.draft_model_runner is not None:
            saved_seq_lens = dict(self.draft_model_runner._seq_lens)
        self._build_next_cache(B, seq_ids)
        if self.draft_model_runner is not None:
            self.draft_model_runner._seq_lens = saved_seq_lens

        if _profile:
            torch.cuda.synchronize(self.device)
            t_cache = time.perf_counter()
            self._profile_count = getattr(self, '_profile_count', 0) + 1
            self._swap_count = getattr(self, '_swap_count', 0) + (1 if used_swap else 0)
            self._jit_count = getattr(self, '_jit_count', 0) + (0 if used_swap else 1)
            if self._profile_count % 50 == 1:
                cache_stats = self.cache.get_stats()
                logger.info(
                    "Disagg draft PROFILE [step %d] B=%d B_miss=%d K=%d swap=%s "
                    "response=%.2fms cache_build=%.2fms total=%.2fms | "
                    "cumulative: swaps=%d jits=%d hit_rate=%.1f%% "
                    "entries=%d",
                    self._profile_count, B, B_miss, self.K,
                    "yes" if used_swap else "no",
                    (t_send - t_recv) * 1000,
                    (t_cache - t_send) * 1000,
                    (t_cache - t_recv) * 1000,
                    self._swap_count, self._jit_count,
                    cache_stats["disagg_cache_hit_rate"] * 100,
                    cache_stats["disagg_cache_entries"],
                )

    def _jit_speculate(
        self,
        seq_ids: torch.Tensor,
        recovery_tokens: torch.Tensor,
        B_miss: int,
        temperatures: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Just-in-time fallback speculation for cache misses.

        When the speculation cache misses, we need to generate tokens
        on the fly. This is the "neural backup" strategy from the paper.

        If the DraftModelRunner is loaded, uses real model inference
        (sequential K-step decode). Otherwise falls back to random tokens.

        Args:
            seq_ids: [B_miss] — sequence IDs for cache misses.
            recovery_tokens: [B_miss] — bonus tokens to start from.
            B_miss: Number of cache misses.
            temperatures: [B_miss] — per-request sampling temperatures.
                None means greedy for all requests.

        Returns:
            tokens: [B_miss, K] — fallback draft tokens.
            logits: [B_miss, K, V] or None — fallback logits.
        """
        # Neural JIT: use DraftModelRunner if available
        if (
            self.draft_model_runner is not None
            and self.draft_model_runner._model_loaded
        ):
            # Compute starting positions from tracked sequence lengths
            positions = torch.tensor(
                [
                    self.draft_model_runner._seq_lens.get(int(sid), 0)
                    for sid in seq_ids.tolist()
                ],
                dtype=torch.long,
                device=self.device,
            )
            tokens, logits = self.draft_model_runner.sequential_speculate(
                recovery_tokens=recovery_tokens,
                positions=positions,
                seq_ids=seq_ids,
                num_steps=self.K,
                temperature=temperatures,
                saguaro_sampler=self.saguaro_sampler if self.saguaro_c is not None else None,
            )
            return tokens, logits

        # Random token fallback (when model not loaded)
        tokens = torch.randint(
            0,
            self.vocab_size,
            (B_miss, self.K),
            device=self.device,
            dtype=torch.int64,
        )
        tokens[:, 0] = recovery_tokens

        logits = torch.zeros(
            B_miss,
            self.K,
            self.vocab_size,
            dtype=self.dtype,
            device=self.device,
        ).uniform_()

        return tokens, logits

    def _build_next_cache(
        self,
        batch_size: int,
        seq_ids: torch.Tensor,
    ) -> None:
        """Pre-compute speculation cache for the NEXT round.

        Called after sending the response to the target. Runs while
        the target is verifying — this is the core async overlap.

        Uses glue decode to get K+1 logits (K from JIT + 1 from glue),
        enabling prediction at K+1 positions including the "all accepted"
        outcome. Tree decode runs K steps for all N = B×(K+1)×F branches
        using dedicated block tables.

        Args:
            batch_size: B, number of sequences.
            seq_ids: [B] — sequence IDs.
        """
        self.cache.reset()

        B = batch_size
        K = self.K
        runner = self.draft_model_runner

        if runner is None or not runner._model_loaded:
            return
        if self._last_draft_tokens is None or self._last_draft_logits is None:
            return

        # Adaptive fan-out: reduce branches at high batch sizes to keep
        # cache build time bounded. Tree decode cost is O(B × (K+1) × F × K).
        # At B>32, reduce F to keep total branches under ~500.
        max_branches = 504  # ~500, divisible by common K+1 values
        F = self.fan_out
        if B * (K + 1) * F > max_branches:
            F = max(1, max_branches // (B * (K + 1)))

        N = B * (K + 1) * F
        # Skip cache build if N is too large — the logits tensor alone
        # would be [N, V] which can OOM on the draft GPU.
        # Budget: ~500 branches max to keep memory under ~250MB for logits.
        if N > max_branches:
            return
        if B * (K + 1) * F > max_branches:
            F = max(1, max_branches // (B * (K + 1)))

        # Recycle dedicated blocks from the previous round's tree decode.
        runner.recycle_dedicated_blocks()

        draft_tokens = self._last_draft_tokens  # [B, K]
        draft_logits = self._last_draft_logits  # [B, K, V]
        rec_tokens = self._last_bonus_tokens    # [B]

        seq_ids_list = seq_ids.tolist()

        _profile = os.environ.get("DISAGG_DRAFT_PROFILE", "0") == "1"
        if _profile:
            torch.cuda.synchronize(self.device)
            _tc0 = time.perf_counter()

        # ----- Step 1: Glue decode for K+1th logits -----
        # Feed the last JIT token to get logits at position K.
        # This writes KV at position base+K in the main blocks, which is
        # safe with block table swapping (swap overwrites those blocks on
        # hit, JIT overwrites on miss).
        glue_logits = runner.glue_decode(
            tokens=draft_tokens[:, -1],
            seq_ids=seq_ids,
        )  # [B, V]

        # Save post-glue _seq_lens so we can restore after tree decode
        post_glue_lens = {
            sid: runner._seq_lens.get(sid, 0) for sid in seq_ids_list
        }

        # ----- Step 2: Predict bonus token candidates at K+1 positions -----
        # Combine JIT logits (K positions) + glue logits (1 position)
        # outcome_logits: [B, K+1, V]
        outcome_logits = torch.cat(
            [draft_logits, glue_logits.unsqueeze(1)], dim=1
        )
        # outcome_tokens: [B, K+1] = [recovery, draft_0, ..., draft_{K-1}]
        outcome_tokens = torch.cat(
            [rec_tokens.unsqueeze(1), draft_tokens], dim=1
        )

        # Mask continuation token at each position and pick top-F
        masked_logits = outcome_logits.clone()  # [B, K+1, V]
        # At positions 0..K-1, mask the NEXT token (continuation)
        masked_logits[:, :-1, :] = masked_logits[:, :-1, :].scatter(
            dim=2,
            index=outcome_tokens[:, 1:].unsqueeze(2),
            value=float("-inf"),
        )
        # Position K: no masking (bonus is standard next-token prediction)
        _, topk_indices = torch.topk(masked_logits, F, dim=-1)  # [B, K+1, F]

        # Flatten into N = B * (K+1) * F cache entries
        Kp1 = K + 1
        # N already computed above
        batch_ids_grid = torch.arange(
            B, device=self.device
        ).view(B, 1, 1).expand(B, Kp1, F)
        k_pos_grid = torch.arange(
            Kp1, device=self.device, dtype=torch.int64
        ).view(1, Kp1, 1).expand(B, Kp1, F)

        k_positions = k_pos_grid.reshape(-1)          # [N]
        bonus_candidates = topk_indices.reshape(-1)    # [N]
        entry_batch_ids = batch_ids_grid.reshape(-1)   # [N]

        if _profile:
            torch.cuda.synchronize(self.device)
            _tc1 = time.perf_counter()

        # ----- Step 2: Bounds check for dedicated blocks -----
        blocks_per_branch = (K + runner.block_size) // runner.block_size + 1
        total_needed = N * blocks_per_branch
        # Check total available blocks: bump headroom + free list.
        available = (runner.num_kv_blocks - runner._next_free_block) + len(runner._free_list)
        if available < total_needed:
            for sid in seq_ids_list:
                if sid in post_glue_lens:
                    runner._seq_lens[sid] = post_glue_lens[sid] - 1
            return

        # Allocate dedicated blocks from free list first, then bump pointer.
        dedicated_blocks = []
        for _ in range(total_needed):
            dedicated_blocks.append(runner._alloc_one_block())
        branch_block_start = dedicated_blocks[0] if dedicated_blocks else 0
        runner.reserve_dedicated_blocks(dedicated_blocks)

        # ----- Step 3: Build per-branch block tables (vectorized) -----
        base_lens_t = torch.tensor(
            [self._round_base_lens.get(int(seq_ids[b].item()), 0)
             for b in range(B)],
            dtype=torch.int64, device=self.device,
        )
        prefix_lens = base_lens_t[entry_batch_ids] + 1 + k_positions  # [N]

        bs = runner.block_size
        M = runner.max_num_blocks

        # Start with parent block tables for each branch (GPU copy)
        seq_ids_for_branches = seq_ids[entry_batch_ids].to(torch.int64)  # [N]
        branch_block_tables = runner._block_table_gpu[
            seq_ids_for_branches
        ].contiguous()  # [N, M]

        # Compute write range per branch
        first_write_blk = prefix_lens // bs  # [N]
        last_write_blk = (prefix_lens + K - 1) // bs  # [N]

        # Build per-branch dedicated block mapping from allocated blocks.
        # dedicated_blocks is a flat list of total_needed blocks.
        # Branch n gets blocks [n*blocks_per_branch : (n+1)*blocks_per_branch].
        ded_tensor = torch.tensor(
            dedicated_blocks, dtype=torch.int64, device=self.device
        ).view(N, blocks_per_branch)  # [N, blocks_per_branch]

        # Replace write-range blocks with dedicated blocks
        j_range = torch.arange(
            blocks_per_branch, device=self.device, dtype=torch.int64
        )
        tbl_indices = first_write_blk.unsqueeze(1) + j_range.unsqueeze(0)
        valid = tbl_indices < M
        n_idx = torch.arange(
            N, device=self.device
        ).unsqueeze(1).expand_as(tbl_indices)
        branch_block_tables[
            n_idx[valid], tbl_indices[valid].to(torch.int64)
        ] = ded_tensor[valid].to(torch.int32)

        # ----- Step 4: Copy KV from parent to dedicated blocks -----
        if _profile:
            torch.cuda.synchronize(self.device)
            _tc2 = time.perf_counter()

        parent_tables = runner._block_table_gpu[
            seq_ids_for_branches
        ]  # [N, M] — original parent tables
        src_indices = tbl_indices.clamp(max=M - 1)
        src_block_ids = parent_tables[
            n_idx, src_indices.to(torch.int64)
        ].to(torch.int64)
        dst_block_ids = ded_tensor  # [N, blocks_per_branch]
        copy_mask = valid & (src_block_ids != dst_block_ids)
        if copy_mask.any() and runner.kv_caches is not None:
            src_flat = src_block_ids[copy_mask]
            dst_flat = dst_block_ids[copy_mask]
            for layer_kv in runner.kv_caches:
                layer_kv[:, dst_flat] = layer_kv[:, src_flat]

        # ----- Step 5: Batched tree decode (K steps) -----
        if _profile:
            torch.cuda.synchronize(self.device)
            _tc3 = time.perf_counter()

        seq_ids_expanded = seq_ids[entry_batch_ids]
        all_tokens = torch.zeros(N, K, dtype=torch.int64, device=self.device)
        all_logits = torch.zeros(
            N, K, self.vocab_size, dtype=self.dtype, device=self.device
        )
        current_ids = bonus_candidates.clone()

        max_prefix = int(prefix_lens.max().item())
        max_context_hint = max_prefix + K + 1

        for depth in range(K):
            positions = prefix_lens + depth
            context_lens = prefix_lens + depth + 1

            logits = runner.tree_decode_step(
                input_ids=current_ids,
                positions=positions,
                seq_lens=context_lens,
                seq_ids_expanded=seq_ids_expanded,
                block_tables=branch_block_tables,
                max_seq_len_hint=max_context_hint,
            )

            all_logits[:, depth] = logits
            next_tokens = logits.argmax(dim=-1)
            all_tokens[:, depth] = next_tokens
            current_ids = next_tokens

        # ----- Step 6: Populate cache -----
        if _profile:
            torch.cuda.synchronize(self.device)
            _tc4 = time.perf_counter()

        self.cache.populate(
            seq_ids=seq_ids[entry_batch_ids],
            k_positions=k_positions,
            bonus_tokens=bonus_candidates,
            draft_tokens=all_tokens,
            draft_logits=all_logits,
            branch_block_tables=branch_block_tables,
            prefix_lens=prefix_lens,
        )

        # Restore _seq_lens to pre-glue values (undo glue's +1).
        # The glue decode advanced _seq_lens by 1 for each sequence.
        # We need to restore so the next round's reset is correct.
        for sid in seq_ids_list:
            if sid in post_glue_lens:
                runner._seq_lens[sid] = post_glue_lens[sid] - 1

        if _profile:
            _tc5 = time.perf_counter()
            self._cache_build_count = getattr(
                self, '_cache_build_count', 0
            ) + 1
            if self._cache_build_count % 100 == 1:
                logger.info(
                    "Disagg draft CACHE_BUILD [%d] B=%d N=%d "
                    "predict=%.2fms blk_tbl=%.2fms kv_copy=%.2fms "
                    "tree=%.2fms pop=%.2fms total=%.2fms",
                    self._cache_build_count, B, N,
                    (_tc1 - _tc0) * 1000,
                    (_tc2 - _tc1) * 1000,
                    (_tc3 - _tc2) * 1000,
                    (_tc4 - _tc3) * 1000,
                    (_tc5 - _tc4) * 1000,
                    (_tc5 - _tc0) * 1000,
                )

    def _handle_free_seq(self) -> None:
        """Handle FREE_SEQ command: release resources for completed sequences.

        Frees KV cache blocks, sequence length tracking, round base
        lengths, and swap state for sequences that have finished.
        """
        seq_ids = self.comm.recv_free_seq()
        freed = 0
        for sid in seq_ids.tolist():
            sid = int(sid)
            self._round_base_lens.pop(sid, None)

            # Clear swap state. Don't call release_owned_blocks here
            # because the owned blocks are already in _block_tables[sid]
            # (swap_block_tables puts them there) and free_blocks will
            # recycle them.
            self._swap_states.pop(sid, None)

            if self.draft_model_runner is not None:
                self.draft_model_runner.free_blocks(sid)
                freed += 1
        if freed:
            logger.debug("Disagg draft freed %d sequences.", freed)

    def _handle_exit(self) -> None:
        """Handle EXIT command: log stats and shutdown."""
        if self._step_times:
            avg_ms = sum(self._step_times) * 1000 / len(self._step_times)
            logger.info(
                "Disagg draft draft worker shutting down. "
                "Avg step time: %.2f ms (%d steps). "
                "Cache hit rate: %.1f%%",
                avg_ms,
                len(self._step_times),
                self.cache.hit_rate * 100,
            )
        else:
            logger.info("Disagg draft draft worker shutting down (no steps executed).")


class DisaggDraftTargetInterface:
    """Target-side interface for communicating with the disagg draft draft worker.

    This class is used by the vLLM model runner / speculator on the
    target side. It sends verification outcomes to the draft worker
    and receives pre-computed speculations.

    This is the counterpart to DisaggDraftWorker — one runs on the target
    GPU(s), the other on the draft GPU.

    Args:
        communicator: NCCL communicator connected to the draft worker.
        num_speculative_tokens: K, the speculation depth.
        vocab_size: Vocabulary size.
        device: Target CUDA device.
        dtype: Data type for logits.
    """

    def __init__(
        self,
        communicator: DisaggDraftCommunicator,
        num_speculative_tokens: int,
        vocab_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.comm = communicator
        self.K = num_speculative_tokens
        self.vocab_size = vocab_size
        self.device = device
        self.dtype = dtype

    def request_prefill(
        self,
        input_ids: torch.Tensor,
        num_tokens: torch.Tensor,
        seq_ids: torch.Tensor | None = None,
    ) -> None:
        """Request the draft worker to prefill new sequences.

        Args:
            input_ids: [total_tokens] — flattened input tokens.
            num_tokens: [B_new] — per-sequence token counts.
            seq_ids: [B_new] — stable sequence IDs (optional).
        """
        self.comm.send_command(DisaggDraftCommand.PREFILL)
        self.comm.send_prefill_data(input_ids, num_tokens, seq_ids=seq_ids)

    def request_speculation(
        self,
        seq_ids: torch.Tensor,
        k_accepted: torch.Tensor,
        bonus_tokens: torch.Tensor,
        batch_size: int,
        temperatures: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Send verification outcome and receive draft tokens (synchronous).

        Args:
            seq_ids: [B] — sequence IDs.
            k_accepted: [B] — tokens accepted per sequence.
            bonus_tokens: [B] — bonus token per sequence.
            batch_size: B.
            temperatures: [B] — per-request sampling temperatures.

        Returns:
            cache_hits: [B] — boolean cache hit mask.
            draft_tokens: [B, K] — speculated tokens.
            draft_logits: [B, K, V] — draft logits (zeros if not cached).
        """
        self.comm.send_command(DisaggDraftCommand.SPECULATE)
        self.comm.send_verification_outcome(
            seq_ids, k_accepted, bonus_tokens,
            temperatures=temperatures,
        )
        return self.comm.recv_speculation(batch_size)

    def request_exit(self) -> None:
        """Request the draft worker to shut down."""
        self.comm.send_command(DisaggDraftCommand.EXIT)

    def request_free_seq(self, seq_ids: torch.Tensor) -> None:
        """Request the draft worker to free completed sequences.

        Args:
            seq_ids: [N] — sequence IDs to free.
        """
        self.comm.send_command(DisaggDraftCommand.FREE_SEQ)
        self.comm.send_free_seq(seq_ids)
