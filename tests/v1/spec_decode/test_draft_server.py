# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DraftServer module."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from vllm.v1.spec_decode.draft_data_models import (
    DraftCommand,
    FreeSeqRequest,
    PrefillRequest,
    TensorRef,
    VerificationOutcome,
    encode_command,
)
from vllm.v1.spec_decode.draft_server import DraftServer


def _make_tensor_ref(**overrides) -> TensorRef:
    defaults = dict(shape=(4,), dtype="int64", buffer_id="buf-0", nbytes=32)
    defaults.update(overrides)
    return TensorRef(**defaults)


def _make_vllm_config() -> MagicMock:
    """Create a minimal mock VllmConfig."""
    cfg = MagicMock()
    cfg.speculative_config = MagicMock()
    cfg.speculative_config.num_speculative_tokens = 5
    cfg.scheduler_config = MagicMock()
    cfg.scheduler_config.max_num_seqs = 32
    cfg.model_config = MagicMock()
    cfg.model_config.dtype = "float16"
    return cfg


@pytest.fixture
def draft_server(tmp_path):
    """Create a DraftServer bound to a random IPC address."""
    addr = f"ipc://{tmp_path}/draft_server_test"
    server = DraftServer(
        vllm_config=_make_vllm_config(),
        bind_address=addr,
    )
    yield server
    server._cleanup()


# ------------------------------------------------------------------
# Request namespacing tests
# ------------------------------------------------------------------


class TestRequestNamespacing:
    """Verify composite key (verify_server_id, seq_id) namespacing."""

    def test_make_key(self, draft_server: DraftServer):
        key = draft_server._make_key("vs-1", 42)
        assert key == ("vs-1", 42)

    def test_register_request_creates_state(self, draft_server: DraftServer):
        key = draft_server._register_request("vs-1", 10)
        assert key in draft_server._request_state
        assert key in draft_server._verify_servers["vs-1"]

    def test_register_same_seq_id_different_servers(
        self, draft_server: DraftServer
    ):
        """Two verify servers can use the same seq_id without collision."""
        k1 = draft_server._register_request("vs-1", 1)
        k2 = draft_server._register_request("vs-2", 1)
        assert k1 != k2
        assert k1 in draft_server._request_state
        assert k2 in draft_server._request_state

    def test_unregister_request_removes_state(
        self, draft_server: DraftServer
    ):
        draft_server._register_request("vs-1", 5)
        draft_server._unregister_request("vs-1", 5)
        assert ("vs-1", 5) not in draft_server._request_state
        # Server entry should be cleaned up when empty
        assert "vs-1" not in draft_server._verify_servers

    def test_unregister_nonexistent_is_safe(
        self, draft_server: DraftServer
    ):
        """Unregistering a key that doesn't exist should not raise."""
        draft_server._unregister_request("vs-99", 999)

    def test_get_request_state_creates_if_absent(
        self, draft_server: DraftServer
    ):
        key = ("vs-1", 7)
        state = draft_server._get_request_state(key)
        assert isinstance(state, dict)
        assert key in draft_server._request_state

    def test_multiple_requests_per_server(
        self, draft_server: DraftServer
    ):
        draft_server._register_request("vs-1", 1)
        draft_server._register_request("vs-1", 2)
        draft_server._register_request("vs-1", 3)
        assert len(draft_server._verify_servers["vs-1"]) == 3

        draft_server._unregister_request("vs-1", 2)
        assert len(draft_server._verify_servers["vs-1"]) == 2
        assert ("vs-1", 2) not in draft_server._request_state


# ------------------------------------------------------------------
# Command dispatch tests
# ------------------------------------------------------------------


class TestCommandDispatch:
    """Verify _dispatch routes commands to the correct handlers."""

    def test_dispatch_speculate(self, draft_server: DraftServer):
        outcome = VerificationOutcome(
            verify_server_id="vs-1",
            batch_size=2,
            seq_ids_ref=_make_tensor_ref(),
            k_accepted_ref=_make_tensor_ref(buffer_id="k"),
            bonus_tokens_ref=_make_tensor_ref(buffer_id="b"),
        )
        cmd_bytes = encode_command("SPECULATE", outcome)
        from vllm.v1.spec_decode.draft_data_models import decode_command

        command = decode_command(cmd_bytes)
        # Should not raise — placeholder handler just logs
        asyncio.run(
            draft_server._dispatch("vs-1", b"vs-1", command)
        )

    def test_dispatch_prefill_registers_request(
        self, draft_server: DraftServer
    ):
        import torch
        from unittest.mock import patch

        prefill = PrefillRequest(
            verify_server_id="vs-1",
            seq_id=42,
            prompt_token_ids_ref=_make_tensor_ref(shape=(128,)),
        )
        cmd_bytes = encode_command("PREFILL", prefill)
        from vllm.v1.spec_decode.draft_data_models import decode_command

        command = decode_command(cmd_bytes)
        # Mock _nccl_recv to return a dummy tensor
        with patch.object(
            draft_server,
            "_nccl_recv",
            return_value=torch.zeros(128, dtype=torch.int64),
        ):
            asyncio.run(
                draft_server._dispatch("vs-1", b"vs-1", command)
            )
        # Prefill should register the request
        assert ("vs-1", 42) in draft_server._request_state

    def test_dispatch_free_seq(self, draft_server: DraftServer):
        import torch
        from unittest.mock import patch

        # Register a request first so free_seq has something to free
        draft_server._register_request("vs-1", 10)
        draft_server._register_request("vs-1", 11)

        free = FreeSeqRequest(
            verify_server_id="vs-1",
            seq_ids_ref=_make_tensor_ref(shape=(2,)),
        )
        cmd_bytes = encode_command("FREE_SEQ", free)
        from vllm.v1.spec_decode.draft_data_models import decode_command

        command = decode_command(cmd_bytes)
        # Mock _nccl_recv to return seq_ids tensor
        with patch.object(
            draft_server,
            "_nccl_recv",
            return_value=torch.tensor([10, 11], dtype=torch.int64),
        ):
            asyncio.run(
                draft_server._dispatch("vs-1", b"vs-1", command)
            )
        # Requests should be unregistered
        assert ("vs-1", 10) not in draft_server._request_state
        assert ("vs-1", 11) not in draft_server._request_state

    def test_dispatch_exit_cleans_up(
        self, draft_server: DraftServer
    ):
        # Register some requests first
        draft_server._register_request("vs-1", 1)
        draft_server._register_request("vs-1", 2)
        assert len(draft_server._verify_servers.get("vs-1", set())) == 2

        cmd_bytes = encode_command("EXIT")
        from vllm.v1.spec_decode.draft_data_models import decode_command

        command = decode_command(cmd_bytes)
        asyncio.run(
            draft_server._dispatch("vs-1", b"vs-1", command)
        )

        # All state for vs-1 should be cleaned up
        assert "vs-1" not in draft_server._verify_servers
        assert ("vs-1", 1) not in draft_server._request_state
        assert ("vs-1", 2) not in draft_server._request_state

    def test_dispatch_unknown_command(
        self, draft_server: DraftServer
    ):
        """Unknown commands should be logged but not raise."""
        command = DraftCommand(command="UNKNOWN_CMD", payload=b"")
        asyncio.run(
            draft_server._dispatch("vs-1", b"vs-1", command)
        )


# ------------------------------------------------------------------
# Cleanup tests
# ------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_clears_state(self, draft_server: DraftServer):
        draft_server._register_request("vs-1", 1)
        draft_server._register_request("vs-2", 2)
        draft_server._cleanup()
        assert len(draft_server._request_state) == 0
        assert len(draft_server._verify_servers) == 0
        assert draft_server._socket is None
        assert draft_server._ctx is None

    def test_double_cleanup_is_safe(self, draft_server: DraftServer):
        draft_server._cleanup()
        draft_server._cleanup()  # Should not raise


# ------------------------------------------------------------------
# Prefill handler tests
# ------------------------------------------------------------------


class TestHandlePrefill:
    """Verify _handle_prefill receives tensors and delegates correctly."""

    def test_prefill_standalone_model(self, draft_server: DraftServer):
        """Standalone draft model: runs standard prefill."""
        import torch
        from unittest.mock import MagicMock, patch

        draft_server.needs_hidden_states = False

        # Set up a mock runner
        runner = MagicMock()
        runner._model_loaded = True
        draft_server.draft_model_runner = runner

        prefill = PrefillRequest(
            verify_server_id="vs-1",
            seq_id=7,
            prompt_token_ids_ref=_make_tensor_ref(shape=(64,)),
        )

        prompt_ids = torch.arange(64, dtype=torch.int64)
        with patch.object(
            draft_server, "_nccl_recv", return_value=prompt_ids
        ):
            asyncio.run(
                draft_server._handle_prefill("vs-1", b"vs-1", prefill)
            )

        # Request should be registered
        assert ("vs-1", 7) in draft_server._request_state
        # Runner.prefill should have been called
        runner.prefill.assert_called_once()
        call_kwargs = runner.prefill.call_args
        assert call_kwargs[1]["input_ids"] is prompt_ids

    def test_prefill_eagle_with_hidden_states(
        self, draft_server: DraftServer
    ):
        """EAGLE method: runs eagle_prefill with hidden states."""
        import torch
        from unittest.mock import MagicMock, patch, call

        draft_server.needs_hidden_states = True

        runner = MagicMock()
        runner._model_loaded = True
        runner.model = MagicMock()
        runner.model.combine_hidden_states = MagicMock(
            side_effect=lambda x: x
        )
        runner._seq_lens = {}
        draft_server.draft_model_runner = runner

        prefill = PrefillRequest(
            verify_server_id="vs-2",
            seq_id=99,
            prompt_token_ids_ref=_make_tensor_ref(shape=(32,)),
            hidden_states_ref=_make_tensor_ref(
                shape=(32, 128), dtype="float16"
            ),
        )

        prompt_ids = torch.arange(32, dtype=torch.int64)
        hidden_states = torch.randn(32, 128, dtype=torch.float16)

        recv_calls = [prompt_ids, hidden_states]
        recv_iter = iter(recv_calls)
        with patch.object(
            draft_server,
            "_nccl_recv",
            side_effect=lambda *a, **kw: next(recv_iter),
        ):
            asyncio.run(
                draft_server._handle_prefill("vs-2", b"vs-2", prefill)
            )

        assert ("vs-2", 99) in draft_server._request_state
        runner.eagle_prefill.assert_called_once()

    def test_prefill_no_runner(self, draft_server: DraftServer):
        """When runner is None, prefill should not crash."""
        import torch
        from unittest.mock import patch

        draft_server.draft_model_runner = None
        draft_server.needs_hidden_states = False

        prefill = PrefillRequest(
            verify_server_id="vs-1",
            seq_id=5,
            prompt_token_ids_ref=_make_tensor_ref(shape=(16,)),
        )

        with patch.object(
            draft_server,
            "_nccl_recv",
            return_value=torch.zeros(16, dtype=torch.int64),
        ):
            asyncio.run(
                draft_server._handle_prefill("vs-1", b"vs-1", prefill)
            )

        # Request should still be registered even if runner is None
        assert ("vs-1", 5) in draft_server._request_state

    def test_prefill_clears_stale_round_state(
        self, draft_server: DraftServer
    ):
        """Prefill should clear stale _round_base_lens and _swap_states."""
        import torch
        from unittest.mock import MagicMock, patch

        draft_server.needs_hidden_states = False
        runner = MagicMock()
        runner._model_loaded = True
        draft_server.draft_model_runner = runner

        # Pre-populate stale state
        draft_server._round_base_lens[3] = 100
        draft_server._swap_states[3] = {"old": True}

        prefill = PrefillRequest(
            verify_server_id="vs-1",
            seq_id=3,
            prompt_token_ids_ref=_make_tensor_ref(shape=(10,)),
        )

        with patch.object(
            draft_server,
            "_nccl_recv",
            return_value=torch.arange(10, dtype=torch.int64),
        ):
            asyncio.run(
                draft_server._handle_prefill("vs-1", b"vs-1", prefill)
            )

        assert 3 not in draft_server._round_base_lens
        assert 3 not in draft_server._swap_states


# ------------------------------------------------------------------
# Free seq handler tests
# ------------------------------------------------------------------


class TestHandleFreeSeq:
    """Verify _handle_free_seq receives tensors and frees resources."""

    def test_free_seq_with_runner(self, draft_server: DraftServer):
        """Free seq should call runner.free_blocks and unregister."""
        import torch
        from unittest.mock import MagicMock, patch

        runner = MagicMock()
        runner._block_tables = {}
        runner._seq_lens = {}
        draft_server.draft_model_runner = runner

        # Register requests first
        draft_server._register_request("vs-1", 10)
        draft_server._register_request("vs-1", 20)

        free_req = FreeSeqRequest(
            verify_server_id="vs-1",
            seq_ids_ref=_make_tensor_ref(shape=(2,)),
        )

        with patch.object(
            draft_server,
            "_nccl_recv",
            return_value=torch.tensor([10, 20], dtype=torch.int64),
        ):
            asyncio.run(
                draft_server._handle_free_seq("vs-1", b"vs-1", free_req)
            )

        # Runner.free_blocks should have been called for each seq
        assert runner.free_blocks.call_count == 2
        # Requests should be unregistered
        assert ("vs-1", 10) not in draft_server._request_state
        assert ("vs-1", 20) not in draft_server._request_state

    def test_free_seq_clears_round_state(self, draft_server: DraftServer):
        """Free seq should clear _round_base_lens and _swap_states."""
        import torch
        from unittest.mock import MagicMock, patch

        runner = MagicMock()
        runner._block_tables = {}
        runner._seq_lens = {}
        draft_server.draft_model_runner = runner

        draft_server._register_request("vs-1", 5)
        draft_server._round_base_lens[5] = 42
        draft_server._swap_states[5] = {"data": True}

        free_req = FreeSeqRequest(
            verify_server_id="vs-1",
            seq_ids_ref=_make_tensor_ref(shape=(1,)),
        )

        with patch.object(
            draft_server,
            "_nccl_recv",
            return_value=torch.tensor([5], dtype=torch.int64),
        ):
            asyncio.run(
                draft_server._handle_free_seq("vs-1", b"vs-1", free_req)
            )

        assert 5 not in draft_server._round_base_lens
        assert 5 not in draft_server._swap_states

    def test_free_seq_no_runner(self, draft_server: DraftServer):
        """When runner is None, free_seq should still unregister."""
        import torch
        from unittest.mock import patch

        draft_server.draft_model_runner = None
        draft_server._register_request("vs-1", 8)

        free_req = FreeSeqRequest(
            verify_server_id="vs-1",
            seq_ids_ref=_make_tensor_ref(shape=(1,)),
        )

        with patch.object(
            draft_server,
            "_nccl_recv",
            return_value=torch.tensor([8], dtype=torch.int64),
        ):
            asyncio.run(
                draft_server._handle_free_seq("vs-1", b"vs-1", free_req)
            )

        assert ("vs-1", 8) not in draft_server._request_state

    def test_free_seq_eagle_prefix_caching(
        self, draft_server: DraftServer
    ):
        """Free seq should cache EAGLE KV blocks before freeing."""
        import torch
        from unittest.mock import MagicMock, patch

        runner = MagicMock()
        runner._block_tables = {15: [100, 101, 102]}
        runner._seq_lens = {15: 48}
        runner._block_table_gpu = MagicMock()
        runner._block_table_gpu.shape = (64,)
        draft_server.draft_model_runner = runner

        # Set up prompt hash for the sequence
        draft_server._seq_prompt_hash[15] = 12345
        draft_server._register_request("vs-1", 15)

        free_req = FreeSeqRequest(
            verify_server_id="vs-1",
            seq_ids_ref=_make_tensor_ref(shape=(1,)),
        )

        with patch.object(
            draft_server,
            "_nccl_recv",
            return_value=torch.tensor([15], dtype=torch.int64),
        ):
            asyncio.run(
                draft_server._handle_free_seq("vs-1", b"vs-1", free_req)
            )

        # EAGLE prefix cache should have the blocks
        assert 12345 in draft_server._eagle_prefix_cache
        cached_blocks, cached_len = draft_server._eagle_prefix_cache[12345]
        assert cached_blocks == [100, 101, 102]
        assert cached_len == 48
        # free_blocks should NOT have been called (blocks owned by cache)
        runner.free_blocks.assert_not_called()


# ------------------------------------------------------------------
# Multi-verify-server batching tests
# ------------------------------------------------------------------


class TestMultiVerifyServerBatching:
    """Verify batching of SPECULATE commands from multiple verify servers."""

    def test_enqueue_speculation_adds_to_pending(
        self, draft_server: DraftServer
    ):
        """_enqueue_speculation should add entries to the pending queue."""
        outcome = VerificationOutcome(
            verify_server_id="vs-1",
            batch_size=1,
            seq_ids_ref=_make_tensor_ref(),
            k_accepted_ref=_make_tensor_ref(buffer_id="k"),
            bonus_tokens_ref=_make_tensor_ref(buffer_id="b"),
        )
        draft_server._enqueue_speculation("vs-1", b"vs-1", outcome)
        assert len(draft_server._pending_speculations) == 1
        assert draft_server._batch_first_arrival is not None

    def test_enqueue_sets_first_arrival_once(
        self, draft_server: DraftServer
    ):
        """First arrival timestamp should only be set on the first enqueue."""
        outcome = VerificationOutcome(
            verify_server_id="vs-1",
            batch_size=1,
            seq_ids_ref=_make_tensor_ref(),
            k_accepted_ref=_make_tensor_ref(buffer_id="k"),
            bonus_tokens_ref=_make_tensor_ref(buffer_id="b"),
        )
        draft_server._enqueue_speculation("vs-1", b"vs-1", outcome)
        first_ts = draft_server._batch_first_arrival

        draft_server._enqueue_speculation("vs-2", b"vs-2", outcome)
        # Timestamp should not change on second enqueue
        assert draft_server._batch_first_arrival == first_ts

    def test_dispatch_speculate_queues_instead_of_processing(
        self, draft_server: DraftServer
    ):
        """SPECULATE commands should be queued, not processed immediately."""
        from unittest.mock import AsyncMock, patch

        outcome = VerificationOutcome(
            verify_server_id="vs-1",
            batch_size=1,
            seq_ids_ref=_make_tensor_ref(),
            k_accepted_ref=_make_tensor_ref(buffer_id="k"),
            bonus_tokens_ref=_make_tensor_ref(buffer_id="b"),
        )
        cmd_bytes = encode_command("SPECULATE", outcome)
        from vllm.v1.spec_decode.draft_data_models import decode_command

        command = decode_command(cmd_bytes)

        with patch.object(
            draft_server,
            "_handle_speculation",
            new_callable=AsyncMock,
        ) as mock_handle:
            asyncio.run(
                draft_server._dispatch("vs-1", b"vs-1", command)
            )
            # Should NOT have called _handle_speculation directly
            mock_handle.assert_not_called()

        # Should be in the pending queue
        assert len(draft_server._pending_speculations) == 1

    def test_dispatch_speculate_triggers_batch_at_max_size(
        self, draft_server: DraftServer
    ):
        """When pending queue reaches max_batch_size, batch is processed."""
        from unittest.mock import AsyncMock, patch

        draft_server._max_batch_size = 2

        outcome = VerificationOutcome(
            verify_server_id="vs-1",
            batch_size=1,
            seq_ids_ref=_make_tensor_ref(),
            k_accepted_ref=_make_tensor_ref(buffer_id="k"),
            bonus_tokens_ref=_make_tensor_ref(buffer_id="b"),
        )
        cmd_bytes = encode_command("SPECULATE", outcome)
        from vllm.v1.spec_decode.draft_data_models import decode_command

        command = decode_command(cmd_bytes)

        with patch.object(
            draft_server,
            "_handle_speculation",
            new_callable=AsyncMock,
        ) as mock_handle:
            # First SPECULATE — should queue, not process
            asyncio.run(
                draft_server._dispatch("vs-1", b"vs-1", command)
            )
            assert len(draft_server._pending_speculations) == 1
            mock_handle.assert_not_called()

            # Second SPECULATE — should trigger batch processing
            asyncio.run(
                draft_server._dispatch("vs-2", b"vs-2", command)
            )
            assert mock_handle.call_count == 2
            assert len(draft_server._pending_speculations) == 0

    def test_dispatch_non_speculate_flushes_pending(
        self, draft_server: DraftServer
    ):
        """Non-SPECULATE commands should flush pending speculations first."""
        import torch
        from unittest.mock import AsyncMock, patch

        outcome = VerificationOutcome(
            verify_server_id="vs-1",
            batch_size=1,
            seq_ids_ref=_make_tensor_ref(),
            k_accepted_ref=_make_tensor_ref(buffer_id="k"),
            bonus_tokens_ref=_make_tensor_ref(buffer_id="b"),
        )
        # Enqueue a speculation
        draft_server._enqueue_speculation("vs-1", b"vs-1", outcome)
        assert len(draft_server._pending_speculations) == 1

        # Now dispatch a PREFILL command
        prefill = PrefillRequest(
            verify_server_id="vs-2",
            seq_id=42,
            prompt_token_ids_ref=_make_tensor_ref(shape=(16,)),
        )
        cmd_bytes = encode_command("PREFILL", prefill)
        from vllm.v1.spec_decode.draft_data_models import decode_command

        command = decode_command(cmd_bytes)

        with patch.object(
            draft_server,
            "_handle_speculation",
            new_callable=AsyncMock,
        ) as mock_spec, patch.object(
            draft_server,
            "_nccl_recv",
            return_value=torch.zeros(16, dtype=torch.int64),
        ):
            asyncio.run(
                draft_server._dispatch("vs-2", b"vs-2", command)
            )
            # Pending speculation should have been flushed
            mock_spec.assert_called_once()
            assert len(draft_server._pending_speculations) == 0

    def test_process_batched_speculation_clears_queue(
        self, draft_server: DraftServer
    ):
        """_process_batched_speculation should clear the pending queue."""
        from unittest.mock import AsyncMock, patch

        outcome = VerificationOutcome(
            verify_server_id="vs-1",
            batch_size=1,
            seq_ids_ref=_make_tensor_ref(),
            k_accepted_ref=_make_tensor_ref(buffer_id="k"),
            bonus_tokens_ref=_make_tensor_ref(buffer_id="b"),
        )
        draft_server._enqueue_speculation("vs-1", b"vs-1", outcome)
        draft_server._enqueue_speculation("vs-2", b"vs-2", outcome)

        with patch.object(
            draft_server,
            "_handle_speculation",
            new_callable=AsyncMock,
        ) as mock_handle:
            asyncio.run(draft_server._process_batched_speculation())
            assert mock_handle.call_count == 2
            assert len(draft_server._pending_speculations) == 0
            assert draft_server._batch_first_arrival is None

    def test_process_batched_speculation_respects_max_batch_size(
        self, draft_server: DraftServer
    ):
        """Only max_batch_size entries should be processed per round."""
        from unittest.mock import AsyncMock, patch

        draft_server._max_batch_size = 2

        outcome = VerificationOutcome(
            verify_server_id="vs-1",
            batch_size=1,
            seq_ids_ref=_make_tensor_ref(),
            k_accepted_ref=_make_tensor_ref(buffer_id="k"),
            bonus_tokens_ref=_make_tensor_ref(buffer_id="b"),
        )
        # Enqueue 3 items
        draft_server._enqueue_speculation("vs-1", b"vs-1", outcome)
        draft_server._enqueue_speculation("vs-2", b"vs-2", outcome)
        draft_server._enqueue_speculation("vs-3", b"vs-3", outcome)

        with patch.object(
            draft_server,
            "_handle_speculation",
            new_callable=AsyncMock,
        ) as mock_handle:
            asyncio.run(draft_server._process_batched_speculation())
            # Only 2 should be processed
            assert mock_handle.call_count == 2
            # 1 should remain
            assert len(draft_server._pending_speculations) == 1
            # Timer should be reset for remaining items
            assert draft_server._batch_first_arrival is not None

    def test_process_batched_speculation_empty_queue_is_noop(
        self, draft_server: DraftServer
    ):
        """Processing an empty queue should be a no-op."""
        asyncio.run(draft_server._process_batched_speculation())
        assert len(draft_server._pending_speculations) == 0

    def test_maybe_flush_batch_no_pending(
        self, draft_server: DraftServer
    ):
        """_maybe_flush_batch with no pending items should be a no-op."""
        asyncio.run(draft_server._maybe_flush_batch())
        assert len(draft_server._pending_speculations) == 0

    def test_maybe_flush_batch_before_timeout(
        self, draft_server: DraftServer
    ):
        """_maybe_flush_batch should not flush before timeout expires."""
        import time
        from unittest.mock import AsyncMock, patch

        outcome = VerificationOutcome(
            verify_server_id="vs-1",
            batch_size=1,
            seq_ids_ref=_make_tensor_ref(),
            k_accepted_ref=_make_tensor_ref(buffer_id="k"),
            bonus_tokens_ref=_make_tensor_ref(buffer_id="b"),
        )
        draft_server._batch_timeout_s = 10.0  # Very long timeout
        draft_server._enqueue_speculation("vs-1", b"vs-1", outcome)

        with patch.object(
            draft_server,
            "_handle_speculation",
            new_callable=AsyncMock,
        ) as mock_handle:
            asyncio.run(draft_server._maybe_flush_batch())
            # Should NOT have flushed — timeout hasn't expired
            mock_handle.assert_not_called()
            assert len(draft_server._pending_speculations) == 1

    def test_maybe_flush_batch_after_timeout(
        self, draft_server: DraftServer
    ):
        """_maybe_flush_batch should flush after timeout expires."""
        import time
        from unittest.mock import AsyncMock, patch

        outcome = VerificationOutcome(
            verify_server_id="vs-1",
            batch_size=1,
            seq_ids_ref=_make_tensor_ref(),
            k_accepted_ref=_make_tensor_ref(buffer_id="k"),
            bonus_tokens_ref=_make_tensor_ref(buffer_id="b"),
        )
        draft_server._batch_timeout_s = 0.0  # Immediate timeout
        draft_server._enqueue_speculation("vs-1", b"vs-1", outcome)
        # Force the first arrival to be in the past
        draft_server._batch_first_arrival = time.monotonic() - 1.0

        with patch.object(
            draft_server,
            "_handle_speculation",
            new_callable=AsyncMock,
        ) as mock_handle:
            asyncio.run(draft_server._maybe_flush_batch())
            mock_handle.assert_called_once()
            assert len(draft_server._pending_speculations) == 0

    def test_cleanup_clears_pending_speculations(
        self, draft_server: DraftServer
    ):
        """_cleanup should clear pending speculations."""
        outcome = VerificationOutcome(
            verify_server_id="vs-1",
            batch_size=1,
            seq_ids_ref=_make_tensor_ref(),
            k_accepted_ref=_make_tensor_ref(buffer_id="k"),
            bonus_tokens_ref=_make_tensor_ref(buffer_id="b"),
        )
        draft_server._enqueue_speculation("vs-1", b"vs-1", outcome)
        assert len(draft_server._pending_speculations) == 1

        draft_server._cleanup()
        assert len(draft_server._pending_speculations) == 0
        assert draft_server._batch_first_arrival is None

    def test_batching_demultiplexes_to_correct_servers(
        self, draft_server: DraftServer
    ):
        """Each entry in the batch should be dispatched to the correct
        verify server with the correct identity and outcome."""
        from unittest.mock import AsyncMock, patch, call

        outcomes = []
        for i, vs_id in enumerate(["vs-1", "vs-2", "vs-3"]):
            outcome = VerificationOutcome(
                verify_server_id=vs_id,
                batch_size=1,
                seq_ids_ref=_make_tensor_ref(buffer_id=f"s-{i}"),
                k_accepted_ref=_make_tensor_ref(buffer_id=f"k-{i}"),
                bonus_tokens_ref=_make_tensor_ref(buffer_id=f"b-{i}"),
            )
            outcomes.append(outcome)
            draft_server._enqueue_speculation(
                vs_id, vs_id.encode(), outcome
            )

        with patch.object(
            draft_server,
            "_handle_speculation",
            new_callable=AsyncMock,
        ) as mock_handle:
            asyncio.run(draft_server._process_batched_speculation())

            assert mock_handle.call_count == 3
            # Verify each call got the correct server id and outcome
            for i, (vs_id, outcome) in enumerate(
                zip(["vs-1", "vs-2", "vs-3"], outcomes)
            ):
                actual_call = mock_handle.call_args_list[i]
                assert actual_call == call(
                    vs_id, vs_id.encode(), outcome
                )

    def test_max_batch_size_from_config(self, draft_server: DraftServer):
        """max_batch_size should be set from the vllm_config."""
        # The fixture creates with max_num_seqs=32
        assert draft_server._max_batch_size == 32


# ------------------------------------------------------------------
# Timeout-based eviction tests
# ------------------------------------------------------------------


class TestTimeoutEviction:
    """Tests for verify-server timeout-based eviction."""

    def test_eviction_timeout_default(self, draft_server: DraftServer):
        """Default eviction timeout should be 30 seconds."""
        assert draft_server._eviction_timeout_s == 30.0

    def test_last_seen_initialized_empty(self, draft_server: DraftServer):
        """_verify_server_last_seen should start empty."""
        assert draft_server._verify_server_last_seen == {}

    def test_dispatch_updates_last_seen(self, draft_server: DraftServer):
        """_dispatch should update _verify_server_last_seen."""
        import time

        cmd = DraftCommand(command="HEALTHCHECK", payload=b"")
        before = time.monotonic()
        asyncio.run(
            draft_server._dispatch("vs-1", b"vs-1", cmd)
        )
        after = time.monotonic()

        assert "vs-1" in draft_server._verify_server_last_seen
        ts = draft_server._verify_server_last_seen["vs-1"]
        assert before <= ts <= after

    def test_check_evictions_no_servers(self, draft_server: DraftServer):
        """_check_evictions should be a no-op with no verify servers."""
        draft_server._check_evictions()
        # No error, no state change

    def test_check_evictions_within_timeout(
        self, draft_server: DraftServer
    ):
        """Servers within timeout should NOT be evicted."""
        import time

        draft_server._register_request("vs-1", 10)
        draft_server._verify_server_last_seen["vs-1"] = time.monotonic()

        draft_server._check_evictions()

        # Server and request should still be present
        assert "vs-1" in draft_server._verify_servers
        assert ("vs-1", 10) in draft_server._request_state

    def test_check_evictions_expired_server(
        self, draft_server: DraftServer
    ):
        """Servers past the timeout should be evicted."""
        import time

        draft_server._register_request("vs-1", 10)
        draft_server._register_request("vs-1", 20)
        # Simulate last seen well in the past
        draft_server._verify_server_last_seen["vs-1"] = (
            time.monotonic() - 60.0
        )

        draft_server._check_evictions()

        # Server and all its requests should be gone
        assert "vs-1" not in draft_server._verify_servers
        assert "vs-1" not in draft_server._verify_server_last_seen
        assert ("vs-1", 10) not in draft_server._request_state
        assert ("vs-1", 20) not in draft_server._request_state

    def test_check_evictions_only_expired_server(
        self, draft_server: DraftServer
    ):
        """Only the timed-out server should be evicted, not others."""
        import time

        now = time.monotonic()
        draft_server._register_request("vs-1", 10)
        draft_server._register_request("vs-2", 20)
        # vs-1 timed out, vs-2 is recent
        draft_server._verify_server_last_seen["vs-1"] = now - 60.0
        draft_server._verify_server_last_seen["vs-2"] = now

        draft_server._check_evictions()

        assert "vs-1" not in draft_server._verify_servers
        assert "vs-2" in draft_server._verify_servers
        assert ("vs-2", 20) in draft_server._request_state

    def test_check_evictions_clears_round_state(
        self, draft_server: DraftServer
    ):
        """Eviction should clear per-round state for evicted seqs."""
        import time

        draft_server._register_request("vs-1", 10)
        draft_server._round_base_lens[10] = 42
        draft_server._swap_states[10] = {"some": "state"}
        draft_server._verify_server_last_seen["vs-1"] = (
            time.monotonic() - 60.0
        )

        draft_server._check_evictions()

        assert 10 not in draft_server._round_base_lens
        assert 10 not in draft_server._swap_states

    def test_check_evictions_frees_kv_blocks(
        self, draft_server: DraftServer
    ):
        """Eviction should call runner.free_blocks for each seq."""
        import time

        mock_runner = MagicMock()
        draft_server.draft_model_runner = mock_runner

        draft_server._register_request("vs-1", 10)
        draft_server._register_request("vs-1", 20)
        draft_server._verify_server_last_seen["vs-1"] = (
            time.monotonic() - 60.0
        )

        draft_server._check_evictions()

        freed_ids = {
            call.args[0] for call in mock_runner.free_blocks.call_args_list
        }
        assert freed_ids == {10, 20}

    def test_check_evictions_removes_pending_speculations(
        self, draft_server: DraftServer
    ):
        """Eviction should remove pending speculations for the server."""
        import time

        outcome = VerificationOutcome(
            verify_server_id="vs-1",
            batch_size=1,
            seq_ids_ref=_make_tensor_ref(),
            k_accepted_ref=_make_tensor_ref(buffer_id="k"),
            bonus_tokens_ref=_make_tensor_ref(buffer_id="b"),
        )
        draft_server._register_request("vs-1", 10)
        draft_server._enqueue_speculation("vs-1", b"vs-1", outcome)
        draft_server._verify_server_last_seen["vs-1"] = (
            time.monotonic() - 60.0
        )

        # Also add a pending speculation from another server
        outcome2 = VerificationOutcome(
            verify_server_id="vs-2",
            batch_size=1,
            seq_ids_ref=_make_tensor_ref(),
            k_accepted_ref=_make_tensor_ref(buffer_id="k2"),
            bonus_tokens_ref=_make_tensor_ref(buffer_id="b2"),
        )
        draft_server._register_request("vs-2", 30)
        draft_server._enqueue_speculation("vs-2", b"vs-2", outcome2)
        draft_server._verify_server_last_seen["vs-2"] = time.monotonic()

        draft_server._check_evictions()

        # vs-1's pending speculation should be removed
        assert len(draft_server._pending_speculations) == 1
        assert draft_server._pending_speculations[0][0] == "vs-2"

    def test_cleanup_clears_last_seen(self, draft_server: DraftServer):
        """_cleanup should clear _verify_server_last_seen."""
        import time

        draft_server._verify_server_last_seen["vs-1"] = time.monotonic()
        draft_server._cleanup()
        assert draft_server._verify_server_last_seen == {}
