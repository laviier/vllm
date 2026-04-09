# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Disaggregated Draft Speculator — Target-side speculator for disagg draft spec decoding (V2 Model Runner).

This class runs on the target GPU (inside GPUModelRunner V2) and acts as a
thin proxy to the disaggregated draft worker running on a separate GPU.
It implements the same interface as EagleSpeculator so the model runner
can call it identically.

Unlike Eagle (which runs the draft model on the same GPU), disagg_draft:
- Does NOT load a draft model on the target GPU
- Does NOT manage attention or KV cache for drafting
- Does NOT use CUDA graphs for drafting
- Instead, sends verification outcomes to the draft worker via NCCL
  and receives pre-computed draft tokens back
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class DisaggDraftSpeculator:
    """Target-side speculator proxy for disagg draft speculation (V2 Model Runner).

    Created by init_speculator() and stored in GPUModelRunner.speculator.
    On the first propose() call, lazily establishes the NCCL PG to the
    draft worker.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.num_speculative_steps = (
            self.speculative_config.num_speculative_tokens
        )
        self.draft_model_config = self.speculative_config.draft_model_config
        self.vocab_size = self.draft_model_config.get_vocab_size()
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.dtype = vllm_config.model_config.dtype

        # Pre-allocate draft token buffer
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )

        # Draft logits for rejection sampling (None = strict mode)
        self.draft_logits: torch.Tensor | None = None
        if self.speculative_config.rejection_sample_method == "probabilistic":
            self.draft_logits = torch.zeros(
                self.max_num_reqs,
                self.num_speculative_steps,
                self.vocab_size,
                dtype=self.dtype,
                device=device,
            )

        # disagg draft does not support multimodal inputs for drafting
        self.supports_mm_inputs = False

        # DisaggDraftTargetInterface — lazily created on first propose()
        self._target_interface = None
        self._nccl_connect_attempted = False

        # Reference to model runner's request states (set via set_req_states)
        self._req_states = None

        # Track per-sequence state for the draft worker
        self._disagg_prefilled_reqs: set[str] = set()
        self._disagg_req_to_seq_id: dict[str, int] = {}
        self._disagg_next_seq_id: int = 0
        self._disagg_free_seq_ids: list[int] = []  # recycled seq_ids

        self._propose_count = 0

        # Determine TP rank — only rank 0 communicates with draft worker.
        try:
            from vllm.distributed.parallel_state import get_tp_group
            self._tp_rank = get_tp_group().rank_in_group
        except Exception:
            self._tp_rank = 0

        logger.info(
            "DisaggDraftSpeculator (V2) created: K=%d, V=%d, device=%s, tp_rank=%d",
            self.num_speculative_steps,
            self.vocab_size,
            device,
            self._tp_rank,
        )

    def set_target_interface(self, interface) -> None:
        """Inject the DisaggDraftTargetInterface after NCCL PG setup."""
        self._target_interface = interface
        logger.info("DisaggDraftSpeculator: target interface connected.")

    def set_req_states(self, req_states) -> None:
        """Store reference to model runner's RequestState for token access."""
        self._req_states = req_states
        # Cache of prompt tokens for new requests, keyed by req_id.
        # Populated by cache_new_request_tokens(), consumed by
        # _prefill_new_requests().
        self._pending_prompt_tokens: dict[str, list[int]] = {}

    def cache_new_request_tokens(
        self, req_id: str, prompt_token_ids: list[int]
    ) -> None:
        """Cache prompt tokens for a new request.

        Called from the model runner's add_requests() so the speculator
        has the actual prompt tokens available when propose() runs later.
        """
        self._pending_prompt_tokens[req_id] = list(prompt_token_ids)

    @property
    def is_connected(self) -> bool:
        return self._target_interface is not None

    def _lazy_connect(self) -> bool:
        """Try to lazily establish the NCCL PG to the draft worker.
        Only TP rank 0 connects; other ranks skip."""
        if self._nccl_connect_attempted:
            return self.is_connected
        self._nccl_connect_attempted = True

        # Only TP rank 0 communicates with the draft worker.
        if self._tp_rank != 0:
            logger.info("Disagg draft: TP rank %d — skipping NCCL connect.", self._tp_rank)
            return False

        nccl_init_method = self.speculative_config.disagg_draft_nccl_init_method
        if not nccl_init_method:
            logger.warning(
                "Disagg draft: disagg_draft_nccl_init_method not set. "
                "Draft worker connection not available."
            )
            return False

        try:
            from datetime import timedelta
            from urllib.parse import urlparse

            import torch.distributed as dist
            from vllm.v1.worker.gpu.spec_decode.disagg_draft.communication import (
                DisaggDraftCommunicator,
            )
            from vllm.v1.worker.gpu.spec_decode.disagg_draft.draft_worker import (
                DisaggDraftTargetInterface,
            )

            logger.info(
                "Disagg draft: Connecting to draft worker via %s", nccl_init_method
            )

            parsed = urlparse(nccl_init_method)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 29500

            store = dist.TCPStore(
                host_name=host,
                port=port,
                world_size=2,
                is_master=False,
                timeout=timedelta(seconds=120),
            )
            disagg_pg = dist.ProcessGroupNCCL(
                store, rank=0, size=2,
                timeout=timedelta(hours=24),
            )

            spec_config = self.speculative_config
            communicator = DisaggDraftCommunicator(
                process_group=disagg_pg,
                peer_rank=1,
                num_speculative_tokens=spec_config.num_speculative_tokens,
                max_batch_size=self.max_num_reqs,
                vocab_size=self.vocab_size,
                device=self.device,
                dtype=self.dtype,
            )

            self._target_interface = DisaggDraftTargetInterface(
                communicator=communicator,
                num_speculative_tokens=spec_config.num_speculative_tokens,
                vocab_size=self.vocab_size,
                device=self.device,
                dtype=self.dtype,
            )

            logger.info("Disagg draft: Target interface connected to draft worker.")
            return True

        except Exception:
            logger.exception("Disagg draft: Failed to connect to draft worker.")
            return False

    # ------------------------------------------------------------------
    # Interface methods matching EagleSpeculator (V2 Model Runner)
    # ------------------------------------------------------------------

    def load_model(self, target_model: nn.Module) -> None:
        """No-op: disagg draft draft model lives on the draft GPU."""
        self.model = _DisaggDraftModelStub()
        logger.info("DisaggDraftSpeculator.load_model: no-op (draft on separate GPU)")

    def set_attn(self, model_state, kv_cache_config,
                 block_tables) -> None:
        """No-op: disagg draft draft model manages its own attention."""
        pass

    def capture_model(self) -> None:
        """No-op: CUDA graphs for draft model are on draft GPU."""
        pass

    @torch.inference_mode()
    def propose(
        self,
        input_batch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Generate draft tokens by communicating with the draft worker.
        Only TP rank 0 does NCCL communication; other ranks return zeros."""
        num_reqs = input_batch.num_reqs
        K = self.num_speculative_steps

        if dummy_run or self._tp_rank != 0:
            return torch.zeros(
                num_reqs, K,
                dtype=torch.int64, device=self.device,
            )

        if not self.is_connected:
            self._lazy_connect()

        if not self.is_connected:
            return torch.zeros(
                num_reqs, K,
                dtype=torch.int64, device=self.device,
            )

        self._propose_count += 1

        # Skip warmup/dummy requests — they pollute the draft worker's
        # state with fake seq_ids and KV cache entries.
        if any(rid.startswith('_warmup_') or rid.startswith('_dummy_')
               for rid in input_batch.req_ids):
            return torch.zeros(
                num_reqs, K,
                dtype=torch.int64, device=self.device,
            )

        try:
            return self._do_propose(
                input_batch, num_sampled, num_rejected, last_sampled,
            )
        except Exception as e:
            logger.warning("Disagg draft propose failed: %s", e, exc_info=True)
            return torch.zeros(
                num_reqs, K,
                dtype=torch.int64, device=self.device,
            )

    def _do_propose(
        self,
        input_batch,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
    ) -> torch.Tensor:
        """Core propose logic: manage sequences and communicate with draft."""
        num_reqs = input_batch.num_reqs
        req_ids = input_batch.req_ids
        idx_mapping = input_batch.idx_mapping[:num_reqs]

        # --- Step 1: Clean up finished requests ---
        active_rids = set(req_ids)
        stale = self._disagg_prefilled_reqs - active_rids
        if stale:
            self._disagg_prefilled_reqs -= stale
            stale_seq_ids = []
            for rid in stale:
                sid = self._disagg_req_to_seq_id.pop(rid, None)
                if sid is not None:
                    stale_seq_ids.append(sid)
                    self._disagg_free_seq_ids.append(sid)
            if stale_seq_ids:
                free_ids = torch.tensor(
                    stale_seq_ids, dtype=torch.int64,
                    device=self.device,
                )
                self._target_interface.request_free_seq(free_ids)

        # --- Step 2: Prefill new requests on the draft worker ---
        new_req_ids = [
            rid for rid in req_ids
            if rid not in self._disagg_prefilled_reqs
        ]
        if new_req_ids:
            self._prefill_new_requests(input_batch, new_req_ids)

        # --- Step 3: Build verification outcome ---
        seq_ids = torch.tensor(
            [self._disagg_req_to_seq_id[rid] for rid in req_ids],
            dtype=torch.int64, device=self.device,
        )

        k_accepted = (num_sampled - 1).clamp(min=0).to(torch.int64)
        bonus_tokens = last_sampled[idx_mapping].squeeze(-1).to(torch.int64)

        # --- Step 4: Request speculation from draft worker ---
        _, draft_toks, draft_logits = (
            self._target_interface.request_speculation(
                seq_ids=seq_ids,
                k_accepted=k_accepted,
                bonus_tokens=bonus_tokens,
                batch_size=num_reqs,
            )
        )

        # Store draft logits if needed for probabilistic rejection sampling
        if self.draft_logits is not None and draft_logits is not None:
            K_actual = min(draft_logits.shape[1], self.draft_logits.shape[1])
            self.draft_logits[:num_reqs, :K_actual] = draft_logits[:, :K_actual]

        self.draft_tokens[:num_reqs] = draft_toks
        return self.draft_tokens[:num_reqs]

    def _prefill_new_requests(self, input_batch, new_req_ids: list[str]):
        """Send prefill data for new requests to the draft worker.

        Uses prompt tokens cached by cache_new_request_tokens() during
        add_requests(), avoiding UVA buffer reads.
        """
        all_prompt_ids = []
        num_tokens_list = []
        new_seq_ids_list = []

        for rid in new_req_ids:
            if rid not in self._disagg_req_to_seq_id:
                # Reuse a freed seq_id if available, otherwise allocate new
                if self._disagg_free_seq_ids:
                    sid = self._disagg_free_seq_ids.pop()
                else:
                    sid = self._disagg_next_seq_id
                    self._disagg_next_seq_id += 1
                self._disagg_req_to_seq_id[rid] = sid

            # Use cached prompt tokens (populated during add_requests)
            prompt_ids = self._pending_prompt_tokens.pop(rid, None)
            if prompt_ids is None or len(prompt_ids) == 0:
                logger.warning(
                    "Disagg draft prefill req %s: no cached prompt tokens, skipping",
                    rid,
                )
                continue

            n_prompt = len(prompt_ids)

            all_prompt_ids.extend(prompt_ids)
            num_tokens_list.append(n_prompt)
            new_seq_ids_list.append(self._disagg_req_to_seq_id[rid])

        if all_prompt_ids:
            input_ids_t = torch.tensor(
                all_prompt_ids, dtype=torch.int64,
                device=self.device,
            )
            num_tokens_t = torch.tensor(
                num_tokens_list, dtype=torch.int32,
                device=self.device,
            )
            new_seq_ids_t = torch.tensor(
                new_seq_ids_list, dtype=torch.int64,
                device=self.device,
            )
            self._target_interface.request_prefill(
                input_ids_t, num_tokens_t,
                seq_ids=new_seq_ids_t,
            )
            self._disagg_prefilled_reqs.update(new_req_ids)


class _DisaggDraftModelStub(nn.Module):
    """Minimal stub so model_runner doesn't crash on speculator.model."""

    def __init__(self):
        super().__init__()
