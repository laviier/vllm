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


class DisaggSpeculatorProxy:
    """Unified target-side proxy for all disaggregated speculation methods.

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

        # Auto-detect whether the method requires hidden state transfer
        self.needs_hidden_states = (
            self.speculative_config.disagg_needs_hidden_states
        )
        self.hidden_size = vllm_config.model_config.get_hidden_size()

        # For EAGLE3, the transfer hidden size is num_aux_layers * target_hidden_size
        # (concatenated aux hidden states). For EAGLE/MTP, it's just hidden_size.
        if self.speculative_config.method == "eagle3":
            # Determine number of aux layers from config or default (3)
            from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
                get_eagle3_aux_layers_from_config,
            )
            aux_layers = get_eagle3_aux_layers_from_config(
                self.speculative_config)
            num_aux = len(aux_layers) if aux_layers else 3
            # Use target_hidden_size from draft config if available,
            # otherwise fall back to base model hidden_size.
            draft_hf = self.speculative_config.draft_model_config.hf_config
            target_hs = getattr(draft_hf, 'target_hidden_size',
                                self.hidden_size)
            self.transfer_hidden_size = num_aux * target_hs
        else:
            self.transfer_hidden_size = self.hidden_size

        # disagg does not support multimodal inputs for drafting
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
            "DisaggSpeculatorProxy created: K=%d, V=%d, device=%s, "
            "tp_rank=%d, needs_hidden_states=%s, hidden_size=%d, "
            "transfer_hidden_size=%d",
            self.num_speculative_steps,
            self.vocab_size,
            device,
            self._tp_rank,
            self.needs_hidden_states,
            self.hidden_size,
            self.transfer_hidden_size,
        )

        if (self.needs_hidden_states
                and vllm_config.cache_config.enable_prefix_caching):
            logger.warning(
                "Disagg EAGLE with prefix caching enabled: acceptance "
                "rate will be degraded for prefix-cached requests because "
                "the EAGLE head cannot access hidden states for cached "
                "prefix tokens. Consider --no-enable-prefix-caching for "
                "best speculation accuracy."
            )

    def set_target_interface(self, interface) -> None:
        """Inject the DisaggDraftTargetInterface after NCCL PG setup."""
        self._target_interface = interface
        logger.info("DisaggSpeculatorProxy: target interface connected.")

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

    def _extract_batch_hidden_states(
        self,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        input_batch,
        num_sampled: torch.Tensor,
        new_req_ids: set | None = None,
        active_indices: list[int] | None = None,
    ) -> torch.Tensor:
        """Extract per-request hidden states from model output.

        Args:
            last_hidden_states: [num_tokens, hidden_size]
            aux_hidden_states: list of [num_tokens, hidden_size] for EAGLE3
            input_batch: input batch with query_start_loc
            num_sampled: [num_reqs] tokens accepted per request
            new_req_ids: set of req_ids that were just prefilled this round
            active_indices: indices into input_batch for active requests.
                If None, uses all requests.
        """
        if active_indices is None:
            indices = list(range(input_batch.num_reqs))
        else:
            indices = active_indices
        B = len(indices)

        last_token_indices = torch.zeros(B, dtype=torch.long, device=last_hidden_states.device)
        for j, orig_i in enumerate(indices):
            rid = input_batch.req_ids[orig_i]
            ns = int(num_sampled[orig_i].item())
            if new_req_ids and rid in new_req_ids:
                # Prefill: use last query token
                last_token_indices[j] = (
                    input_batch.query_start_loc[orig_i + 1] - 1
                )
            else:
                # Decode: use last accepted position
                last_token_indices[j] = (
                    input_batch.query_start_loc[orig_i] + ns - 1
                )

        if aux_hidden_states is not None and len(aux_hidden_states) > 0:
            combined = torch.cat(list(aux_hidden_states), dim=-1)
            return combined[last_token_indices]

        return last_hidden_states[last_token_indices]

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

        nccl_init_method = self.speculative_config.disagg_nccl_init_method
        if not nccl_init_method:
            logger.warning(
                "Disagg draft: disagg_nccl_init_method not set. "
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
                needs_hidden_states=self.needs_hidden_states,
                hidden_size=self.transfer_hidden_size,
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
        logger.info("DisaggSpeculatorProxy.load_model: no-op (draft on separate GPU)")

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
                temperature, last_hidden_states, aux_hidden_states,
            )
        except Exception as e:
            logger.error("Disagg draft propose FAILED: %s", e, exc_info=True)
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
        temperature: torch.Tensor,
        last_hidden_states: torch.Tensor | None = None,
        aux_hidden_states: list[torch.Tensor] | None = None,
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
        # Only prefill requests that have completed their prompt
        # processing (num_sampled > 0). With chunked prefill, a
        # request may appear in the batch before all prompt tokens
        # are processed. We must wait until the full prompt is done.
        new_req_ids = []
        for i, rid in enumerate(req_ids):
            if rid not in self._disagg_prefilled_reqs:
                # Check if this request has finished prefill
                # (num_sampled > 0 means at least one token was sampled)
                if int(num_sampled[i].item()) > 0:
                    new_req_ids.append(rid)
        if new_req_ids:
            self._prefill_new_requests(
                input_batch, new_req_ids,
                last_hidden_states=last_hidden_states,
                aux_hidden_states=aux_hidden_states,
            )

        # --- Step 3: Build verification outcome ---
        # Only include requests that have been prefilled on the draft
        # worker. Requests still in chunked prefill (num_sampled=0)
        # don't have a seq_id yet.
        active_req_indices = [
            i for i, rid in enumerate(req_ids)
            if rid in self._disagg_req_to_seq_id
        ]
        if not active_req_indices:
            return torch.zeros(
                num_reqs, K,
                dtype=torch.int64, device=self.device,
            )

        active_req_ids = [req_ids[i] for i in active_req_indices]
        seq_ids = torch.tensor(
            [self._disagg_req_to_seq_id[rid] for rid in active_req_ids],
            dtype=torch.int64, device=self.device,
        )
        active_idx = torch.tensor(
            active_req_indices, dtype=torch.int64, device=self.device)

        k_accepted = (num_sampled[active_idx] - 1).clamp(min=0).to(torch.int64)
        bonus_tokens = last_sampled[
            idx_mapping[active_idx]
        ].squeeze(-1).to(torch.int64)

        # --- Step 3b: Build extend data for glue decode ---
        # The extend data contains the target's hidden states for the
        # tokens accepted in the CURRENT verification. These are sent
        # to the draft worker so it can run glue decode to fill KV
        # cache gaps.
        #
        # EAGLE conditioning shift: token at position p gets conditioning
        # from hidden state at position p-1. So for n_ext accepted draft
        # tokens, we need:
        #   - Token IDs: the accepted draft tokens (positions 1..n_ext)
        #   - Hidden states: positions 0..n_ext-1 (shifted by 1)
        #     Position 0 = recovery token's hs → conditions first draft
        #     Position 1 = first draft's hs → conditions second draft
        #     etc.
        K = self.num_speculative_steps
        B_active = len(active_req_indices)
        extend_counts = None
        extend_hidden_states = None
        extend_token_ids = None
        if self.needs_hidden_states and aux_hidden_states is not None:
            extend_counts = torch.zeros(
                B_active, dtype=torch.int64, device=self.device)
            extend_hidden_states = torch.zeros(
                B_active, K, self.transfer_hidden_size,
                dtype=self.dtype, device=self.device)
            extend_token_ids = torch.zeros(
                B_active, K, dtype=torch.int64, device=self.device)
            combined = torch.cat(list(aux_hidden_states), dim=-1)
            for j, orig_i in enumerate(active_req_indices):
                k_acc = int(k_accepted[j].item())
                n_ext = min(k_acc, K)
                if n_ext > 0:
                    start = int(
                        input_batch.query_start_loc[orig_i].item())
                    extend_hidden_states[j, :n_ext] = (
                        combined[start:start + n_ext])
                    extend_token_ids[j, :n_ext] = (
                        input_batch.input_ids[
                            start + 1:start + 1 + n_ext
                        ].to(torch.int64))
                    extend_counts[j] = n_ext

        # --- Step 4: Request speculation from draft worker ---
        temps = temperature[idx_mapping[active_idx]].to(torch.float32)

        # Extract per-request hidden states for EAGLE/EAGLE3/MTP methods
        hs = None
        if self.needs_hidden_states and last_hidden_states is not None:
            hs = self._extract_batch_hidden_states(
                last_hidden_states, aux_hidden_states, input_batch,
                num_sampled,
                new_req_ids=set(new_req_ids) if new_req_ids else None,
                active_indices=active_req_indices,
            )

        import time as _time
        _t0 = _time.perf_counter()
        _, draft_toks, draft_logits = \
            self._target_interface.request_speculation(
                seq_ids=seq_ids,
                k_accepted=k_accepted,
                bonus_tokens=bonus_tokens,
                batch_size=B_active,
                temperatures=temps,
                hidden_states=hs,
                extend_counts=extend_counts,
                extend_hidden_states=extend_hidden_states,
                extend_token_ids=extend_token_ids,
            )
        _dt = (_time.perf_counter() - _t0) * 1000
        if not hasattr(self, '_spec_times'):
            self._spec_times = []
        self._spec_times.append(_dt)
        if len(self._spec_times) % 200 == 0:
            avg = sum(self._spec_times[-200:]) / 200
            logger.info(
                "Disagg SPECULATE latency: avg=%.2fms over last 200 calls "
                "(total %d calls)",
                avg, len(self._spec_times),
            )

        # Map draft tokens back to the full batch.
        # Requests not in active_req_indices get zeros (no draft tokens).
        self.draft_tokens[:num_reqs].zero_()
        if self.draft_logits is not None and draft_logits is not None:
            K_actual = min(draft_logits.shape[1], self.draft_logits.shape[1])
            self.draft_logits[:num_reqs].zero_()
            for j, orig_i in enumerate(active_req_indices):
                self.draft_logits[orig_i, :K_actual] = draft_logits[j, :K_actual]

        target_vocab = self.vllm_config.model_config.get_vocab_size()
        draft_toks = draft_toks.clamp(min=0, max=target_vocab - 1)
        for j, orig_i in enumerate(active_req_indices):
            self.draft_tokens[orig_i] = draft_toks[j]

        return self.draft_tokens[:num_reqs]

    def _prefill_new_requests(
        self,
        input_batch,
        new_req_ids: list[str],
        last_hidden_states: torch.Tensor | None = None,
        aux_hidden_states: list[torch.Tensor] | None = None,
    ):
        """Send prefill data for new requests to the draft worker.

        Uses prompt tokens cached by cache_new_request_tokens() during
        add_requests(), avoiding UVA buffer reads.

        When ``needs_hidden_states`` is True and ``last_hidden_states``
        is provided, the target's last hidden state for each new
        request's final prompt token is extracted and sent alongside
        the prompt token IDs so the EAGLE head can begin speculation
        immediately after prefill.
        """
        all_prompt_ids = []
        num_tokens_list = []
        new_seq_ids_list = []
        new_req_batch_indices: list[int] = []

        for rid in new_req_ids:
            if rid not in self._disagg_req_to_seq_id:
                if self._disagg_free_seq_ids:
                    sid = self._disagg_free_seq_ids.pop()
                else:
                    sid = self._disagg_next_seq_id
                    self._disagg_next_seq_id += 1
                self._disagg_req_to_seq_id[rid] = sid

            prompt_ids = self._pending_prompt_tokens.pop(rid, None)
            if prompt_ids is None or len(prompt_ids) == 0:
                logger.warning(
                    "Disagg draft prefill req %s: no cached prompt tokens, skipping",
                    rid,
                )
                continue

            n_prompt = len(prompt_ids)

            # Always send ALL prompt token IDs (needed for draft prefill).
            # Hidden states may be partial (suffix only) if prefix caching
            # is active. The draft worker handles the mismatch.
            all_prompt_ids.extend(prompt_ids)
            num_tokens_list.append(n_prompt)
            new_seq_ids_list.append(self._disagg_req_to_seq_id[rid])

            try:
                batch_idx = list(input_batch.req_ids).index(rid)
                new_req_batch_indices.append(batch_idx)
            except ValueError:
                new_req_batch_indices.append(-1)

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

            # Extract hidden states for ALL prompt tokens of each new
            # request when the method requires them.  The standard EAGLE
            # speculator processes all prompt tokens through the EAGLE
            # head in a single "prefill" step, populating the EAGLE
            # head's KV cache for the full prompt context.  We replicate
            # this by sending all prompt hidden states to the draft
            # worker so it can run a full EAGLE prefill.
            hs = None
            if (self.needs_hidden_states
                    and last_hidden_states is not None
                    and new_req_batch_indices):
                hs_parts = []
                for idx, bi in enumerate(new_req_batch_indices):
                    if bi < 0:
                        continue
                    start = int(input_batch.query_start_loc[bi].item())
                    end = int(input_batch.query_start_loc[bi + 1].item())

                    if (aux_hidden_states is not None
                            and len(aux_hidden_states) > 0):
                        combined = torch.cat(
                            list(aux_hidden_states), dim=-1)
                        hs_parts.append(combined[start:end])
                    else:
                        hs_parts.append(last_hidden_states[start:end])
                if hs_parts:
                    hs = torch.cat(hs_parts, dim=0)

            try:
                self._target_interface.request_prefill(
                    input_ids_t, num_tokens_t,
                    seq_ids=new_seq_ids_t,
                    hidden_states=hs,
                )
            except Exception as e:
                logger.error(
                    "Disagg draft prefill send FAILED: %s", e,
                    exc_info=True,
                )
                # The PREFILL command was already sent. We must send
                # dummy prefill data so the draft worker doesn't hang.
                # Send a minimal 1-token prefill.
                try:
                    dummy_ids = torch.zeros(1, dtype=torch.int64,
                                            device=self.device)
                    dummy_ntok = torch.ones(1, dtype=torch.int32,
                                            device=self.device)
                    dummy_sids = new_seq_ids_t[:1]
                    self._target_interface.comm.send_prefill_data(
                        dummy_ids, dummy_ntok,
                        seq_ids=dummy_sids,
                    )
                except Exception:
                    pass
                return
            self._disagg_prefilled_reqs.update(new_req_ids)


class _DisaggDraftModelStub(nn.Module):
    """Minimal stub so model_runner doesn't crash on speculator.model."""

    def __init__(self):
        super().__init__()
