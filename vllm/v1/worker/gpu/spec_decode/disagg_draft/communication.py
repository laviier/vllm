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
    ):
        self.pg = process_group
        self.peer_rank = peer_rank
        self.K = num_speculative_tokens
        self.max_batch_size = max_batch_size
        self.vocab_size = vocab_size
        self.device = device
        self.dtype = dtype

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

        # Pre-allocate metadata buffer: [batch_size, K, fan_out]
        self._meta_buf = torch.zeros(3, dtype=torch.int64, device=device)

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
    ) -> None:
        """Send verification results to the draft worker.

        Called by target after rejection sampling completes.
        This is non-blocking from the target's perspective — the NCCL
        send completes quickly and the draft worker processes async.

        Args:
            seq_ids: [B] — sequence IDs.
            k_accepted: [B] — number of tokens accepted per sequence.
            bonus_tokens: [B] — bonus token sampled per sequence.
            temperatures: [B] — per-request sampling temperatures
                (float32). None means greedy (all zeros sent).
        """
        B = seq_ids.shape[0]
        self._verify_outcome_buf[:B, 0] = seq_ids
        self._verify_outcome_buf[:B, 1] = k_accepted
        self._verify_outcome_buf[:B, 2] = bonus_tokens

        # Send metadata: [B, K, has_temperatures]
        has_temps = 1 if temperatures is not None else 0
        self._meta_buf[0] = B
        self._meta_buf[1] = self.K
        self._meta_buf[2] = has_temps
        self._pg_send(self._meta_buf)

        # Send fused outcomes
        payload = self._verify_outcome_buf[:B].reshape(-1)  # [B*3]
        self._pg_send(payload)

        # Send per-request temperatures
        if temperatures is not None:
            temps = temperatures[:B].to(
                dtype=torch.float32, device=self.device
            ).contiguous()
            self._pg_send(temps)

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
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Receive verification results from the target worker.

        Called by draft worker at the start of each speculation step.

        Returns:
            batch_size: B, the number of sequences.
            seq_ids: [B] — sequence IDs.
            k_accepted: [B] — acceptance positions.
            bonus_tokens: [B] — bonus tokens.
            temperatures: [B] float32 or None — per-request temperatures.
        """
        # Receive metadata
        self._pg_recv(self._meta_buf)
        B = int(self._meta_buf[0].item())
        has_temps = int(self._meta_buf[2].item())

        # Receive fused outcomes: [B*3]
        payload = torch.empty(B * 3, dtype=torch.int64, device=self.device)
        self._pg_recv(payload)
        outcomes = payload.view(B, 3)

        seq_ids = outcomes[:, 0]
        k_accepted = outcomes[:, 1]
        bonus_tokens = outcomes[:, 2]

        # Receive per-request temperatures if sent
        temperatures = None
        if has_temps:
            temperatures = torch.empty(
                B, dtype=torch.float32, device=self.device
            )
            self._pg_recv(temperatures)

        return B, seq_ids, k_accepted, bonus_tokens, temperatures

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
    ) -> None:
        """Send prefill data for new sequences to the draft worker.

        Called by target when new requests join the batch and need
        their prefix processed by the draft model.

        Args:
            input_ids: [total_tokens] — flattened input tokens for all sequences.
            num_tokens: [B] — number of tokens per sequence.
            seq_ids: [B] — stable sequence IDs for each new request.
                     If None, draft worker uses arange(B) internally.
        """
        B = num_tokens.shape[0]
        total = input_ids.shape[0]

        # Send prefill metadata: [total_tokens, B, has_seq_ids, 0, 0]
        has_seq_ids = 1 if seq_ids is not None else 0
        meta = torch.tensor(
            [total, B, has_seq_ids, 0, 0], dtype=torch.int64, device=self.device
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

    def recv_prefill_data(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Receive prefill data from the target worker.

        Called by draft worker when a PREFILL command is received.

        Returns:
            input_ids: [total_tokens] — flattened input tokens.
            num_tokens: [B] — number of tokens per sequence.
            seq_ids: [B] or None — stable sequence IDs if provided.
        """
        # Receive metadata
        meta = torch.empty(5, dtype=torch.int64, device=self.device)
        self._pg_recv(meta)
        total = int(meta[0].item())
        B = int(meta[1].item())
        has_seq_ids = int(meta[2].item())

        # Receive fused payload
        payload_len = total + B + (B if has_seq_ids else 0)
        payload = torch.empty(payload_len, dtype=torch.int64, device=self.device)
        self._pg_recv(payload)

        input_ids = payload[:total]
        num_tokens = payload[total:total + B]
        seq_ids = payload[total + B:] if has_seq_ids else None

        return input_ids, num_tokens, seq_ids

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
