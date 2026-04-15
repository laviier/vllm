# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NCCL Communication Protocol for disagg draft Disaggregated Draft Worker.

Handles bidirectional communication between the target model (TP group)
and the draft model (separate GPU) using torch.distributed NCCL primitives.

Communication is lightweight — POC measured negligible overhead:
  - Target→Draft: verification outcome (k_accepted, bonus_token) per sequence
    Payload: ~8 bytes per request × batch_size
  - Draft→Target: speculated tokens + logits per sequence
    Payload: K × (token_id + logits) per request × batch_size

All sends/receives use pre-allocated buffers to avoid CUDA malloc overhead.
Tensors are packed into fused int64 payloads following the disagg draft reference impl
pattern for minimal NCCL round trips.

Reference: SSD ref impl ssd/utils/async_helpers/nccl_pack.py
"""

from __future__ import annotations

from enum import IntEnum

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)


class DisaggDraftCommand(IntEnum):
    """Commands sent from target to draft worker."""

    SPECULATE = 0  # Run speculation (normal step)
    PREFILL = 1  # Run prefill for new sequences
    EXIT = 2  # Shutdown draft worker
    FREE_SEQ = 3  # Free completed sequences on draft worker


class DisaggDraftCommunicator:
    """Manages NCCL communication between target and draft workers.

    The communicator is instantiated on both sides (target and draft).
    On the target side, it sends verification outcomes and receives
    draft speculations. On the draft side, it's the reverse.

    All tensors are pre-allocated on the correct device to avoid
    per-step CUDA allocations.

    Uses direct ProcessGroupNCCL.send()/recv() methods instead of
    dist.send()/dist.recv() because the standalone PG may not be
    registered in PyTorch's global group map.

    Args:
        process_group: NCCL process group connecting target and draft.
        peer_rank: Rank of the remote peer (draft rank if on target, vice versa).
        num_speculative_tokens: K, the speculation depth.
        max_batch_size: Maximum batch size.
        vocab_size: Vocabulary size (for logit tensors).
        device: Local CUDA device.
        dtype: Data type for logit communication (default: bfloat16).
        needs_hidden_states: Whether the method requires hidden state
            transfer (True for EAGLE/EAGLE3/MTP, False for standalone).
        hidden_size: Hidden dimension of the target model. Required when
            needs_hidden_states is True.
        needs_aux_hidden_states: Whether the method requires auxiliary
            hidden state transfer (True for EAGLE3 which uses
            intermediate layer hidden states).
        aux_hidden_size: Total size of concatenated auxiliary hidden
            states (num_layers * hidden_size). Required when
            needs_aux_hidden_states is True.
    """

    def __init__(
        self,
        process_group: dist.ProcessGroup,
        peer_rank: int,
        num_speculative_tokens: int,
        max_batch_size: int,
        vocab_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        needs_hidden_states: bool = False,
        hidden_size: int = 0,
        needs_aux_hidden_states: bool = False,
        aux_hidden_size: int = 0,
    ):
        self.pg = process_group
        self.peer_rank = peer_rank
        self.K = num_speculative_tokens
        self.max_batch_size = max_batch_size
        self.vocab_size = vocab_size
        self.device = device
        self.dtype = dtype
        self.needs_hidden_states = needs_hidden_states
        self.hidden_size = hidden_size
        self.needs_aux_hidden_states = needs_aux_hidden_states
        self.aux_hidden_size = aux_hidden_size

        # Whether to send draft logits over NCCL (needed for probabilistic
        # rejection sampling; off by default for strict/greedy).
        self.send_draft_logits = False

        # Pre-allocate command buffer (single int64)
        self._cmd_buf = torch.zeros(1, dtype=torch.int64, device=device)

        # Pre-allocate verification outcome buffers (target→draft)
        # Format: [B, 3] where columns are (seq_id, k_accepted, bonus_token)
        self._verify_outcome_buf = torch.zeros(
            max_batch_size, 3, dtype=torch.int64, device=device
        )

        # Pre-allocate metadata buffer: [batch_size, K, has_temps, has_aux_hs]
        self._meta_buf = torch.zeros(4, dtype=torch.int64, device=device)

        # Pre-allocate response buffers (draft→target)
        # Fused response: [B (cache_hits) + B*K (tokens)] as int64
        self._fused_response_buf = torch.zeros(
            max_batch_size + max_batch_size * self.K,
            dtype=torch.int64,
            device=device,
        )

        # Logits buffer: [B, K, V]
        # NOTE: This is large. For K=7, V=128000, B=16: ~27 GB in bf16.
        # We lazily allocate only what's needed per step.
        self._logits_buf: torch.Tensor | None = None
        self._logits_buf_size = 0

        # Pre-allocate hidden state buffer for EAGLE/EAGLE3/MTP methods
        self._hidden_state_buf: torch.Tensor | None = None
        if self.needs_hidden_states:
            self._hidden_state_buf = torch.zeros(
                max_batch_size,
                hidden_size,
                dtype=dtype,
                device=device,
            )

        # Pre-allocate aux hidden state buffer for EAGLE3
        # Shape: [max_batch_size, aux_hidden_size] where
        # aux_hidden_size = num_layers * hidden_size
        self._aux_hidden_state_buf: torch.Tensor | None = None
        if self.needs_aux_hidden_states:
            self._aux_hidden_state_buf = torch.zeros(
                max_batch_size,
                aux_hidden_size,
                dtype=dtype,
                device=device,
            )

    def _pg_send(self, tensor: torch.Tensor) -> None:
        """Send tensor via direct PG call (works with unregistered PGs)."""
        self.pg.send([tensor.contiguous()], self.peer_rank, 0).wait()

    def _pg_recv(self, tensor: torch.Tensor) -> None:
        """Recv tensor via direct PG call (works with unregistered PGs)."""
        self.pg.recv([tensor], self.peer_rank, 0).wait()

    def _ensure_logits_buf(self, batch_size: int) -> torch.Tensor:
        """Lazily allocate logits buffer for the current batch size."""
        if self._logits_buf is None or batch_size > self._logits_buf_size:
            self._logits_buf_size = batch_size
            self._logits_buf = torch.zeros(
                batch_size,
                self.K,
                self.vocab_size,
                dtype=self.dtype,
                device=self.device,
            )
        return self._logits_buf

    # ---------------------------------------------------------------
    # Target-side methods (called by target worker)
    # ---------------------------------------------------------------

    def send_command(self, cmd: DisaggDraftCommand) -> None:
        """Send a command to the draft worker."""
        self._cmd_buf[0] = cmd.value
        self._pg_send(self._cmd_buf)

    def send_verification_outcome(
        self,
        seq_ids: torch.Tensor,
        k_accepted: torch.Tensor,
        bonus_tokens: torch.Tensor,
        temperatures: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        aux_hidden_states: torch.Tensor | None = None,
        extend_counts: torch.Tensor | None = None,
        extend_hidden_states: torch.Tensor | None = None,
        extend_token_ids: torch.Tensor | None = None,
    ) -> None:
        """Send verification results to the draft worker.

        When extend data is provided (for glue decode), it is sent
        after the main hidden states. The extend data contains:
        - extend_counts: [B] number of accepted draft tokens per seq
        - extend_hidden_states: [B, K, hs] hidden states for accepted tokens
        - extend_token_ids: [B, K] token IDs for accepted tokens
        """
        B = seq_ids.shape[0]
        self._verify_outcome_buf[:B, 0] = seq_ids
        self._verify_outcome_buf[:B, 1] = k_accepted
        self._verify_outcome_buf[:B, 2] = bonus_tokens

        has_temps = 1 if temperatures is not None else 0
        has_aux_hs = (1 if self.needs_aux_hidden_states
                      and aux_hidden_states is not None else 0)
        has_extend = 1 if extend_counts is not None else 0
        self._meta_buf[0] = B
        self._meta_buf[1] = self.K
        self._meta_buf[2] = has_temps
        self._meta_buf[3] = has_aux_hs + (has_extend << 1)
        self._pg_send(self._meta_buf)

        payload = self._verify_outcome_buf[:B].reshape(-1)
        self._pg_send(payload)

        if temperatures is not None:
            self._pg_send(temperatures[:B].to(
                dtype=torch.float32, device=self.device).contiguous())

        if self.needs_hidden_states and hidden_states is not None:
            self._pg_send(hidden_states[:B].contiguous())

        if has_aux_hs:
            self._pg_send(aux_hidden_states[:B].contiguous())

        if has_extend:
            self._pg_send(extend_counts[:B].to(
                torch.int64).contiguous())
            self._pg_send(extend_hidden_states[:B].to(
                self.dtype).contiguous())
            self._pg_send(extend_token_ids[:B].to(
                torch.int64).contiguous())

    def recv_speculation(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Receive pre-computed speculation from the draft worker.

        Args:
            batch_size: Expected batch size B.

        Returns:
            cache_hits: [B] — boolean indicating cache hit per sequence.
            draft_tokens: [B, K] — speculated draft tokens.
            draft_logits: [B, K, V] — draft logits (zeros if not transferred).
        """
        B = batch_size
        fused_len = B + B * self.K

        # Receive fused response: [cache_hits(B) | tokens(B*K)]
        fused = self._fused_response_buf[:fused_len]
        self._pg_recv(fused)

        cache_hits = fused[:B].bool()
        draft_tokens = fused[B:].view(B, self.K)

        logits_buf = self._ensure_logits_buf(B)
        if self.send_draft_logits:
            self._pg_recv(logits_buf[:B])

        return cache_hits, draft_tokens, logits_buf[:B]

    # ---------------------------------------------------------------
    # Draft-side methods (called by draft worker)
    # ---------------------------------------------------------------

    def recv_command(self) -> DisaggDraftCommand:
        """Receive a command from the target worker."""
        self._pg_recv(self._cmd_buf)
        return DisaggDraftCommand(self._cmd_buf[0].item())

    def recv_verification_outcome(
        self,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor | None, torch.Tensor | None,
               torch.Tensor | None, torch.Tensor | None,
               torch.Tensor | None, torch.Tensor | None]:
        """Receive verification results from the target worker.

        Returns:
            Tuple of (B, seq_ids, k_accepted, bonus_tokens, temperatures,
                       hidden_states, aux_hidden_states,
                       extend_counts, extend_hidden_states, extend_token_ids)
        """
        self._pg_recv(self._meta_buf)
        B = int(self._meta_buf[0].item())
        has_temps = int(self._meta_buf[2].item())
        flags = int(self._meta_buf[3].item())
        has_aux_hs = flags & 1
        has_extend = (flags >> 1) & 1

        payload = torch.empty(B * 3, dtype=torch.int64, device=self.device)
        self._pg_recv(payload)
        outcomes = payload.view(B, 3)
        seq_ids = outcomes[:, 0]
        k_accepted = outcomes[:, 1]
        bonus_tokens = outcomes[:, 2]

        temperatures = None
        if has_temps:
            temperatures = torch.empty(
                B, dtype=torch.float32, device=self.device)
            self._pg_recv(temperatures)

        hidden_states = None
        if self.needs_hidden_states:
            self._pg_recv(self._hidden_state_buf[:B])
            hidden_states = self._hidden_state_buf[:B]

        aux_hidden_states = None
        if has_aux_hs:
            assert self._aux_hidden_state_buf is not None
            self._pg_recv(self._aux_hidden_state_buf[:B])
            aux_hidden_states = self._aux_hidden_state_buf[:B]

        extend_counts = None
        extend_hidden_states = None
        extend_token_ids = None
        if has_extend:
            extend_counts = torch.empty(
                B, dtype=torch.int64, device=self.device)
            self._pg_recv(extend_counts)
            extend_hidden_states = torch.empty(
                B, self.K, self.hidden_size,
                dtype=self.dtype, device=self.device)
            self._pg_recv(extend_hidden_states)
            extend_token_ids = torch.empty(
                B, self.K, dtype=torch.int64, device=self.device)
            self._pg_recv(extend_token_ids)

        return (B, seq_ids, k_accepted, bonus_tokens, temperatures,
                hidden_states, aux_hidden_states,
                extend_counts, extend_hidden_states, extend_token_ids)

    def send_speculation(
        self,
        cache_hits: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> None:
        """Send pre-computed speculation to the target worker.

        Sends tokens always. Logits are sent only when
        send_draft_logits is True (needed for probabilistic rejection
        sampling). Skipping logits saves ~1.5MB NCCL transfer per round.

        Args:
            cache_hits: [B] — boolean indicating cache hit per sequence.
            draft_tokens: [B, K] — speculated tokens.
            draft_logits: [B, K, V] — draft model logits.
        """
        B = cache_hits.shape[0]

        # Build fused response: [cache_hits(B) | tokens(B*K)]
        fused_len = B + B * self.K
        fused = self._fused_response_buf[:fused_len]
        fused[:B] = cache_hits.to(torch.int64)
        fused[B:] = draft_tokens.reshape(-1).to(torch.int64)
        self._pg_send(fused)

        # Send logits only if configured (probabilistic rejection needs them)
        if self.send_draft_logits:
            self._pg_send(draft_logits[:B, :self.K].contiguous())

    # ---------------------------------------------------------------
    # Prefill communication
    # ---------------------------------------------------------------

    def send_prefill_data(
        self,
        input_ids: torch.Tensor,
        num_tokens: torch.Tensor,
        seq_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        aux_hidden_states: torch.Tensor | None = None,
    ) -> None:
        """Send prefill data for new sequences to the draft worker.

        Called by target when new requests join the batch and need
        their prefix processed by the draft model.

        Args:
            input_ids: [total_tokens] — flattened input tokens for all sequences.
            num_tokens: [B] — number of tokens per sequence.
            seq_ids: [B] — stable sequence IDs for each new request.
                     If None, draft worker uses arange(B) internally.
            hidden_states: [total_tokens, hidden_size] — target model
                hidden states for ALL prompt tokens. Sent when
                needs_hidden_states is True.
            aux_hidden_states: [total_tokens, aux_hidden_size] —
                concatenated intermediate layer hidden states for
                EAGLE3. Only sent when needs_aux_hidden_states is True.
        """
        B = num_tokens.shape[0]
        total = input_ids.shape[0]

        # Send prefill metadata:
        # [total_tokens, B, has_seq_ids, has_hidden_states, has_aux_hs,
        #  hs_num_tokens]
        # hs_num_tokens is the actual number of hidden state rows,
        # which may be less than total_tokens when prefix caching
        # is active (only suffix tokens have hidden states).
        has_seq_ids = 1 if seq_ids is not None else 0
        has_hs = (1 if self.needs_hidden_states and hidden_states is not None
                  else 0)
        has_aux_hs = (1 if self.needs_aux_hidden_states
                      and aux_hidden_states is not None else 0)
        hs_num_tokens = hidden_states.shape[0] if has_hs else 0
        meta = torch.tensor(
            [total, B, has_seq_ids, has_hs, has_aux_hs, hs_num_tokens],
            dtype=torch.int64, device=self.device,
        )
        self._pg_send(meta)

        # Send fused payload: input_ids + num_tokens [+ seq_ids]
        payload_len = total + B + (B if seq_ids is not None else 0)
        payload = torch.empty(payload_len, dtype=torch.int64, device=self.device)
        payload[:total] = input_ids.to(torch.int64)
        payload[total:total + B] = num_tokens.to(torch.int64)
        if seq_ids is not None:
            payload[total + B:] = seq_ids.to(torch.int64)
        self._pg_send(payload)

        # Send hidden states for EAGLE/EAGLE3/MTP methods
        if has_hs:
            self._pg_send(hidden_states.contiguous())

        # Send aux hidden states for EAGLE3
        if has_aux_hs:
            self._pg_send(aux_hidden_states.contiguous())

    def recv_prefill_data(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None,
               torch.Tensor | None, torch.Tensor | None]:
        """Receive prefill data from the target worker.

        Called by draft worker when a PREFILL command is received.

        Returns:
            input_ids: [total_tokens] — flattened input tokens.
            num_tokens: [B] — number of tokens per sequence.
            seq_ids: [B] or None — stable sequence IDs if provided.
            hidden_states: [total_tokens, hidden_size] or None — target
                hidden states for all prompt tokens.
            aux_hidden_states: [total_tokens, aux_hidden_size] or None —
                concatenated intermediate layer hidden states for EAGLE3.
        """
        # Receive metadata
        meta = torch.empty(6, dtype=torch.int64, device=self.device)
        self._pg_recv(meta)
        total = int(meta[0].item())
        B = int(meta[1].item())
        has_seq_ids = int(meta[2].item())
        has_hidden_states = int(meta[3].item())
        has_aux_hs = int(meta[4].item())
        hs_num_tokens = int(meta[5].item())

        # Receive fused payload
        payload_len = total + B + (B if has_seq_ids else 0)
        payload = torch.empty(payload_len, dtype=torch.int64, device=self.device)
        self._pg_recv(payload)

        input_ids = payload[:total]
        num_tokens = payload[total:total + B]
        seq_ids = payload[total + B:] if has_seq_ids else None

        # Receive hidden states if sent.
        # hs_num_tokens may be less than total when prefix caching
        # is active (only suffix tokens have hidden states).
        hidden_states = None
        if has_hidden_states:
            hs_count = hs_num_tokens if hs_num_tokens > 0 else total
            hs_buf = torch.empty(
                hs_count, self.hidden_size,
                dtype=self.dtype, device=self.device,
            )
            self._pg_recv(hs_buf)
            hidden_states = hs_buf

        # Receive aux hidden states if sent
        aux_hidden_states = None
        if has_aux_hs:
            aux_buf = torch.empty(
                total, self.aux_hidden_size,
                dtype=self.dtype, device=self.device,
            )
            self._pg_recv(aux_buf)
            aux_hidden_states = aux_buf

        return input_ids, num_tokens, seq_ids, hidden_states, aux_hidden_states

    # ---------------------------------------------------------------
    # Sequence lifecycle
    # ---------------------------------------------------------------

    def send_free_seq(self, seq_ids: torch.Tensor) -> None:
        """Tell the draft worker to free resources for completed sequences.

        Args:
            seq_ids: [N] — sequence IDs to free.
        """
        N = seq_ids.shape[0]
        meta = torch.tensor([N], dtype=torch.int64, device=self.device)
        self._pg_send(meta)
        self._pg_send(seq_ids.to(torch.int64).contiguous())

    def recv_free_seq(self) -> torch.Tensor:
        """Receive sequence IDs to free from the target.

        Returns:
            seq_ids: [N] — sequence IDs to free.
        """
        meta = torch.empty(1, dtype=torch.int64, device=self.device)
        self._pg_recv(meta)
        N = int(meta[0].item())
        seq_ids = torch.empty(N, dtype=torch.int64, device=self.device)
        self._pg_recv(seq_ids)
        return seq_ids
