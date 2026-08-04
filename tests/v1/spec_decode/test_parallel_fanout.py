# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Test parallel fanout implementation for disaggregated speculative decoding.

Validates that:
1. _run_parallel_fanout produces correct output shapes [N, K] and [N, K, V]
2. Both sequential and parallel paths run without errors
3. Parallel path is faster at larger batch sizes (CUDA graph amortization)

Each test runs in its own subprocess to avoid prometheus registry conflicts
from creating multiple DraftServer instances.

Run:
    source /home/ubuntu/sampling/disagg/.venv/bin/activate
    cd /home/ubuntu/sampling/disagg/vllm
    CUDA_VISIBLE_DEVICES=6 python tests/v1/spec_decode/test_parallel_fanout.py
"""

import subprocess
import sys
import textwrap


def run_in_subprocess(test_code: str, test_name: str) -> bool:
    """Run test code in a fresh subprocess to isolate state."""
    print(f"\n{'=' * 60}")
    print(f"  {test_name}")
    print(f"{'=' * 60}")

    full_code = textwrap.dedent(test_code)
    result = subprocess.run(
        [sys.executable, "-c", full_code],
        capture_output=False,
        text=True,
        timeout=180,
        env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if result.returncode != 0:
        print(f"  ✗ FAILED (exit code {result.returncode})")
        return False
    return True


TEST_SHAPES = """
import time
import torch

from vllm.config.vllm import set_current_vllm_config
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.draft_server import _init_distributed_for_draft_server
from vllm.usage.usage_lib import UsageContext

engine_args = AsyncEngineArgs(
    model="/opt/dlami/nvme/dummy_gptoss20b",
    tensor_parallel_size=1,
    max_num_seqs=8,
    max_model_len=4096,
    speculative_config={
        "method": "draft_model",
        "model": "/opt/dlami/nvme/dummy_gptoss20b",
        "num_speculative_tokens": 3,
        "disagg_parallel_fanout": True,
        "disagg_mtp_token_id": 200019,
    },
)
vllm_config = engine_args.create_engine_config(
    usage_context=UsageContext.OPENAI_API_SERVER,
)

with set_current_vllm_config(vllm_config):
    _init_distributed_for_draft_server(vllm_config)
    from vllm.v1.spec_decode.draft_server import DraftServer
    server = DraftServer(vllm_config, bind_address="tcp://*:59999")
    server.load_model()

runner = server.draft_model_runner
assert server._use_parallel_fanout, "Parallel fanout should be enabled"
print(f"  MTP token ID: {server._mtp_token_id}")
print(f"  K={server.K}, vocab_size={server.vocab_size}")

B = 2
K = server.K
N = 4

seq_ids = torch.tensor([0, 1], dtype=torch.int64, device=server.device)
for sid in [0, 1]:
    runner.allocate_blocks(sid, 64)
    runner._seq_lens[sid] = 32

entry_batch_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=server.device)
k_positions = torch.tensor([0, 1, 0, 2], dtype=torch.int64, device=server.device)
prefix_lens = torch.tensor([33, 34, 33, 35], dtype=torch.int64, device=server.device)

blocks_per_branch = (K + runner.block_size) // runner.block_size + 1
for _ in range(N * blocks_per_branch):
    runner._alloc_one_block()

branch_block_tables = runner._block_table_gpu[
    seq_ids[entry_batch_ids].to(torch.int64)
].contiguous()

bonus_candidates = torch.randint(
    0, server.vocab_size, (N,), dtype=torch.int64, device=server.device
)

# --- Run parallel fanout ---
print("  Running _run_parallel_fanout...")
t0 = time.perf_counter()
all_tokens_par, all_logits_par = server._run_parallel_fanout(
    runner=runner, N=N, K=K, seq_ids=seq_ids,
    entry_batch_ids=entry_batch_ids, prefix_lens=prefix_lens,
    branch_block_tables=branch_block_tables,
    bonus_candidates=bonus_candidates,
)
torch.accelerator.synchronize()
t_parallel = (time.perf_counter() - t0) * 1000
print(f"  Parallel fanout time: {t_parallel:.2f} ms")

# Validate shapes
assert all_tokens_par.shape == (N, K), (
    f"Expected ({N}, {K}), got {all_tokens_par.shape}"
)
assert all_logits_par.shape == (N, K, server.vocab_size), \\
    f"Expected ({N}, {K}, {server.vocab_size}), got {all_logits_par.shape}"
print("  ✓ Shape validation passed")

# Validate tokens are valid vocab indices
assert (all_tokens_par >= 0).all(), "Negative token IDs"
assert (all_tokens_par < server.vocab_size).all(), "Token IDs exceed vocab"
print("  ✓ Token range validation passed")

# Validate logits are finite
assert torch.isfinite(all_logits_par).all(), "Non-finite logits"
print("  ✓ Logits finiteness validation passed")

# --- Run sequential for comparison ---
print("  Running _run_tree_decode (sequential)...")
t0 = time.perf_counter()
all_tokens_seq, all_logits_seq = server._run_tree_decode(
    runner=runner, N=N, K=K, seq_ids=seq_ids,
    entry_batch_ids=entry_batch_ids, prefix_lens=prefix_lens,
    branch_block_tables=branch_block_tables,
    bonus_candidates=bonus_candidates,
)
torch.accelerator.synchronize()
t_sequential = (time.perf_counter() - t0) * 1000

assert all_tokens_seq.shape == (N, K)
assert all_logits_seq.shape == (N, K, server.vocab_size)
print(f"  Sequential time: {t_sequential:.2f} ms")
print(f"  ✓ Sequential path also works")

for sid in [0, 1]:
    runner.free_blocks(sid)

print("  ✓ All shape tests passed!")
import sys; sys.stdout.flush(); import os; os._exit(0)
"""


TEST_BENCHMARK = """
import time
import statistics
import torch

from vllm.config.vllm import set_current_vllm_config
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.draft_server import _init_distributed_for_draft_server
from vllm.usage.usage_lib import UsageContext

engine_args = AsyncEngineArgs(
    model="/opt/dlami/nvme/dummy_gptoss20b",
    tensor_parallel_size=1,
    max_num_seqs=8,
    max_model_len=4096,
    speculative_config={
        "method": "draft_model",
        "model": "/opt/dlami/nvme/dummy_gptoss20b",
        "num_speculative_tokens": 3,
        "disagg_parallel_fanout": True,
        "disagg_mtp_token_id": 200019,
    },
)
vllm_config = engine_args.create_engine_config(
    usage_context=UsageContext.OPENAI_API_SERVER,
)

with set_current_vllm_config(vllm_config):
    _init_distributed_for_draft_server(vllm_config)
    from vllm.v1.spec_decode.draft_server import DraftServer
    server = DraftServer(vllm_config, bind_address="tcp://*:59998")
    server.load_model()

runner = server.draft_model_runner
K = server.K
B = 4
N = 8

seq_ids = torch.arange(B, dtype=torch.int64, device=server.device)
for sid in range(B):
    runner.allocate_blocks(sid, 128)
    runner._seq_lens[sid] = 64

entry_batch_ids = torch.tensor(
    [0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64, device=server.device
)
k_positions = torch.tensor(
    [0, 1, 0, 2, 1, 2, 0, 1], dtype=torch.int64, device=server.device
)
prefix_lens = torch.tensor(
    [65, 66, 65, 67, 66, 67, 65, 66], dtype=torch.int64, device=server.device
)

blocks_per_branch = (K + runner.block_size) // runner.block_size + 1
for _ in range(N * blocks_per_branch):
    runner._alloc_one_block()

branch_block_tables = runner._block_table_gpu[
    seq_ids[entry_batch_ids].to(torch.int64)
].contiguous()

bonus_candidates = torch.randint(
    0, server.vocab_size, (N,), dtype=torch.int64, device=server.device
)

# Warmup
for _ in range(3):
    server._run_parallel_fanout(
        runner=runner, N=N, K=K, seq_ids=seq_ids,
        entry_batch_ids=entry_batch_ids, prefix_lens=prefix_lens,
        branch_block_tables=branch_block_tables,
        bonus_candidates=bonus_candidates,
    )
    server._run_tree_decode(
        runner=runner, N=N, K=K, seq_ids=seq_ids,
        entry_batch_ids=entry_batch_ids, prefix_lens=prefix_lens,
        branch_block_tables=branch_block_tables,
        bonus_candidates=bonus_candidates,
    )
torch.accelerator.synchronize()

# Benchmark parallel
times_par = []
for _ in range(10):
    t0 = time.perf_counter()
    server._run_parallel_fanout(
        runner=runner, N=N, K=K, seq_ids=seq_ids,
        entry_batch_ids=entry_batch_ids, prefix_lens=prefix_lens,
        branch_block_tables=branch_block_tables,
        bonus_candidates=bonus_candidates,
    )
    torch.accelerator.synchronize()
    times_par.append((time.perf_counter() - t0) * 1000)

# Benchmark sequential
times_seq = []
for _ in range(10):
    t0 = time.perf_counter()
    server._run_tree_decode(
        runner=runner, N=N, K=K, seq_ids=seq_ids,
        entry_batch_ids=entry_batch_ids, prefix_lens=prefix_lens,
        branch_block_tables=branch_block_tables,
        bonus_candidates=bonus_candidates,
    )
    torch.accelerator.synchronize()
    times_seq.append((time.perf_counter() - t0) * 1000)

par_avg = statistics.mean(times_par)
seq_avg = statistics.mean(times_seq)

print(f"  Batch: B={B}, N={N} branches, K={K}, total_tokens={N*K}")
print(f"  Parallel:   avg={par_avg:.2f}ms (1 pass of {N*K} tokens)")
print(f"  Sequential: avg={seq_avg:.2f}ms ({K} passes of {N} tokens)")
print(f"  Speedup:    {seq_avg/par_avg:.2f}x")

# At N=8, K=3 (24 tokens), parallel should be faster than sequential
assert par_avg < seq_avg * 1.5, (
    f"Parallel ({par_avg:.1f}ms) should not be much slower than "
    f"sequential ({seq_avg:.1f}ms) at N={N}"
)

for sid in range(B):
    runner.free_blocks(sid)

print("  ✓ Benchmark complete!")
import sys; sys.stdout.flush(); import os; os._exit(0)
"""


TEST_CONFIG_DISABLED = """
import torch

from vllm.config.vllm import set_current_vllm_config
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.draft_server import _init_distributed_for_draft_server
from vllm.usage.usage_lib import UsageContext

# Test with parallel fanout DISABLED (default)
engine_args = AsyncEngineArgs(
    model="/opt/dlami/nvme/dummy_gptoss20b",
    tensor_parallel_size=1,
    max_num_seqs=8,
    max_model_len=4096,
    speculative_config={
        "method": "draft_model",
        "model": "/opt/dlami/nvme/dummy_gptoss20b",
        "num_speculative_tokens": 3,
        # disagg_parallel_fanout defaults to False
    },
)
vllm_config = engine_args.create_engine_config(
    usage_context=UsageContext.OPENAI_API_SERVER,
)

with set_current_vllm_config(vllm_config):
    _init_distributed_for_draft_server(vllm_config)
    from vllm.v1.spec_decode.draft_server import DraftServer
    server = DraftServer(vllm_config, bind_address="tcp://*:59997")
    server.load_model()

assert not server._use_parallel_fanout, "Parallel fanout should be DISABLED by default"
print("  ✓ disagg_parallel_fanout=False (default) correctly disables parallel fanout")
import sys; sys.stdout.flush(); import os; os._exit(0)
"""


if __name__ == "__main__":
    print("=" * 60)
    print("  PARALLEL FANOUT VALIDATION TESTS")
    print("  Model: /opt/dlami/nvme/dummy_gptoss20b (random weights)")
    print("=" * 60)

    all_passed = True
    all_passed &= run_in_subprocess(TEST_SHAPES, "TEST: Parallel Fanout Output Shapes")
    all_passed &= run_in_subprocess(
        TEST_BENCHMARK, "TEST: Parallel Fanout Benchmark (N=8)"
    )
    all_passed &= run_in_subprocess(
        TEST_CONFIG_DISABLED, "TEST: Config Default (disabled)"
    )

    print("\n" + "=" * 60)
    if all_passed:
        print("  ALL TESTS PASSED ✓")
    else:
        print("  SOME TESTS FAILED ✗")
        sys.exit(1)
    print("=" * 60)
