# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Disaggregated Draft Executor Extension for MultiprocExecutor.

Launches the disaggregated draft worker as a separate process on a
dedicated GPU. Sets up an NCCL process group between the target
(rank 0 of the TP group) and the draft worker for verification
outcome / speculation exchange.

Usage:
    The disagg draft executor is activated when `speculative_config.use_disagg_draft()` is True.
    It extends MultiprocExecutor by overriding `_post_init_executor()` to
    spawn the draft worker process after the TP group is initialized.

Architecture:
    ┌─────────────────────────────────────────┐
    │  MultiprocExecutor                      │
    │  ├── WorkerProc[0] (TP rank 0, GPU 0)  │
    │  ├── WorkerProc[1] (TP rank 1, GPU 1)  │
    │  ├── WorkerProc[2] (TP rank 2, GPU 2)  │
    │  ├── WorkerProc[3] (TP rank 3, GPU 3)  │
    │  └── disagg draft Draft Worker (GPU 4) ← NEW    │
    │       └── NCCL PG ↔ WorkerProc[0]      │
    └─────────────────────────────────────────┘

Reference: SSD paper §3, disagg draft ref impl ssd/engine/draft_runner.py
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import threading
from multiprocessing.connection import Connection
from typing import Any

import torch
import torch.distributed as dist

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import get_loopback_ip, get_open_port
from vllm.utils.system_utils import get_mp_context, set_process_title

logger = init_logger(__name__)


def launch_disagg_draft_worker(
    vllm_config: VllmConfig,
    draft_device_id: int,
    nccl_init_method: str,
    ready_pipe: Connection,
    death_pipe: Connection,
) -> None:
    """Entry point for the disagg draft draft worker process.

    This function runs in a separate process on the draft GPU. It:
    1. Initializes the CUDA device
    2. Sets up an NCCL process group with the target (rank 0)
    3. Creates the DisaggDraftWorker with its communicator
    4. Signals readiness to the parent
    5. Enters the main draft worker loop

    Args:
        vllm_config: Full vLLM configuration.
        draft_device_id: CUDA device ID for the draft model (e.g., 4).
        nccl_init_method: torch.distributed init method (e.g., tcp://...).
        ready_pipe: Pipe to signal readiness to parent.
        death_pipe: Pipe to detect parent exit.
    """
    # Handle signals for graceful shutdown
    shutdown_requested = threading.Event()

    def signal_handler(signum, frame):
        if not shutdown_requested.is_set():
            shutdown_requested.set()
            # Force exit — the main thread may be blocked on NCCL recv
            os._exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        set_process_title(name="DisaggDraft-Worker")
        device = torch.device(f"cuda:{draft_device_id}")
        torch.cuda.set_device(device)

        logger.info(
            "Disagg draft draft worker starting on device %s (PID=%d)",
            device,
            os.getpid(),
        )

        spec_config = vllm_config.speculative_config
        assert spec_config is not None

        # Initialize minimal parallel state for the draft worker process.
        # vLLM models (e.g., Llama) call get_pp_group()/get_tp_group() during
        # __init__, so we need these groups even for a standalone TP=1 PP=1 draft.
        from vllm.config import set_current_vllm_config
        from vllm.distributed.parallel_state import (
            initialize_model_parallel,
            init_distributed_environment,
        )
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://{get_loopback_ip()}:{get_open_port()}",
            local_rank=0,
            backend="nccl",
        )
        with set_current_vllm_config(vllm_config):
            initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
            )

        logger.info("Disagg draft draft worker: parallel state initialized (TP=1, PP=1).")

        # Load the draft model FIRST (before NCCL, so we can signal READY
        # without waiting for the target side to be ready for rendezvous).
        # Keep the vllm_config context active during model loading since
        # model __init__ and weight loading need it.
        from vllm.v1.worker.gpu.spec_decode.disagg_draft.draft_model_runner import (
            DraftModelRunner,
        )

        # Use draft-substituted config for model loading so the model
        # loader sees the draft model's hidden_size/num_layers, not the target's.
        # Also disable torch.compile and CUDA graphs for the draft model
        # since our simplified runner doesn't provide the full forward
        # context that the compiled/captured graph expects.
        from copy import deepcopy
        from vllm.config.compilation import (
            CompilationConfig, CompilationMode, CUDAGraphMode,
        )
        draft_vllm_config = deepcopy(vllm_config)
        draft_vllm_config.model_config = spec_config.draft_model_config
        draft_vllm_config.compilation_config = CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
            custom_ops=["all"],
        )
        # Use TP=1 parallel config for the draft model since it runs
        # on a single GPU. Without this, the model loader would shard
        # weights for TP=4 (the target's config), producing incorrect
        # predictions on a single GPU.
        if spec_config.draft_parallel_config is not None:
            draft_vllm_config.parallel_config = spec_config.draft_parallel_config
        else:
            draft_vllm_config.parallel_config = deepcopy(vllm_config.parallel_config)
            draft_vllm_config.parallel_config.tensor_parallel_size = 1
            draft_vllm_config.parallel_config.pipeline_parallel_size = 1

        with set_current_vllm_config(draft_vllm_config):
            draft_model_runner = DraftModelRunner(
                vllm_config=vllm_config,
                device=device,
            )
            draft_model_runner.load_model()

        # Signal readiness BEFORE NCCL init to avoid deadlock.
        # The NCCL PG will be initialized lazily when the target side
        # is ready to rendezvous (after DisaggDraftSpeculator._lazy_connect()).
        ready_pipe.send({"status": "READY", "nccl_init_method": nccl_init_method})
        ready_pipe.close()
        ready_pipe = None

        logger.info(
            "Disagg draft draft worker ready (model loaded). "
            "Waiting for target to initiate NCCL PG..."
        )

        # Now initialize NCCL PG for disagg draft communication.
        # The default PG was already created by init_distributed_environment,
        # so we must destroy it first.
        dist.destroy_process_group()

        # Create a standalone TCPStore + ProcessGroupNCCL.
        # The target side (_lazy_connect) creates the master TCPStore on
        # the same host:port, so this rendezvous completes the pair.
        from datetime import timedelta
        from urllib.parse import urlparse

        nccl_timeout = timedelta(seconds=600)
        try:
            parsed = urlparse(nccl_init_method)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 29500

            store = dist.TCPStore(
                host_name=host,
                port=port,
                world_size=2,
                is_master=True,  # draft starts first → hosts the store
                timeout=nccl_timeout,
            )
            disagg_pg = dist.ProcessGroupNCCL(
                store, rank=1, size=2,
                timeout=timedelta(hours=24),
            )
        except Exception as e:
            logger.warning(
                "Disagg draft draft worker: NCCL PG init timed out (%s). "
                "Target-side _lazy_connect() was not called yet. "
                "Draft worker will idle until E2E integration is wired. "
                "Error: %s",
                nccl_timeout,
                e,
            )
            # Stay alive so the parent doesn't see an unexpected exit,
            # but don't enter the main loop since we have no PG.
            death_pipe.recv()  # blocks until parent closes
            return

        logger.info("Disagg draft draft worker: NCCL process group initialized.")

        # Create communicator
        from vllm.v1.worker.gpu.spec_decode.disagg_draft.communication import DisaggDraftCommunicator

        communicator = DisaggDraftCommunicator(
            process_group=disagg_pg,
            peer_rank=0,
            num_speculative_tokens=spec_config.num_speculative_tokens,
            max_batch_size=vllm_config.scheduler_config.max_num_seqs,
            vocab_size=spec_config.draft_model_config.get_vocab_size(),
            device=device,
            dtype=vllm_config.model_config.dtype,
        )

        # Create the draft worker
        from vllm.v1.worker.gpu.spec_decode.disagg_draft.draft_worker import DisaggDraftWorker

        draft_worker = DisaggDraftWorker(
            vllm_config=vllm_config,
            device=device,
            communicator=communicator,
        )
        draft_worker.draft_model_runner = draft_model_runner

        logger.info("Disagg draft draft worker entering main loop.")

        # Monitor death pipe in background
        def death_monitor():
            try:
                death_pipe.recv()
            except EOFError:
                pass
            logger.info("Disagg draft draft worker: parent exited, forcing shutdown.")
            shutdown_requested.set()
            # Force exit since the main thread is blocked on NCCL recv
            import os
            os._exit(0)

        threading.Thread(
            target=death_monitor, daemon=True, name="DisaggDraft-DeathMonitor"
        ).start()

        # Enter the main event loop
        draft_worker.run_loop()

    except SystemExit:
        logger.info("Disagg draft draft worker terminated.")
    except Exception:
        logger.exception("Disagg draft draft worker failed.")
        try:
            ready_pipe.send({"status": "FAILED"})
        except Exception:
            pass
    finally:
        if ready_pipe is not None:
            try:
                ready_pipe.close()
            except Exception:
                pass
        if death_pipe is not None:
            try:
                death_pipe.close()
            except Exception:
                pass
        # Clean up distributed
        if dist.is_initialized():
            dist.destroy_process_group()


class DisaggDraftWorkerHandle:
    """Handle to the running disagg draft draft worker process.

    Held by the MultiprocExecutor (or its disagg draft extension) to manage
    the draft worker lifecycle and communicate with it from the
    target side.
    """

    def __init__(
        self,
        proc: multiprocessing.Process,
        death_writer: Connection,
        nccl_init_method: str,
    ):
        self.proc = proc
        self.death_writer = death_writer
        self.nccl_init_method = nccl_init_method
        self._target_pg: dist.ProcessGroup | None = None
        self._target_interface = None

    def init_target_side_pg(
        self,
        device: torch.device,
    ) -> dist.ProcessGroup:
        """Initialize the target-side NCCL process group.

        Must be called from the target worker (rank 0) AFTER the draft
        worker has started its process group initialization. This creates
        the matching rank-0 endpoint.

        Args:
            device: Target CUDA device (e.g., cuda:0).

        Returns:
            The NCCL process group connecting target and draft.
        """
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="nccl",
            init_method=self.nccl_init_method,
            world_size=2,
            rank=0,  # target is rank 0
        )
        self._target_pg = dist.group.WORLD
        return self._target_pg

    def create_target_interface(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """Create the DisaggDraftTargetInterface for target-side communication.

        Args:
            vllm_config: Full vLLM configuration.
            device: Target CUDA device.

        Returns:
            DisaggDraftTargetInterface instance.
        """
        from vllm.v1.worker.gpu.spec_decode.disagg_draft.communication import DisaggDraftCommunicator
        from vllm.v1.worker.gpu.spec_decode.disagg_draft.draft_worker import DisaggDraftTargetInterface

        assert self._target_pg is not None, (
            "Must call init_target_side_pg() before create_target_interface()"
        )

        spec_config = vllm_config.speculative_config
        assert spec_config is not None

        communicator = DisaggDraftCommunicator(
            process_group=self._target_pg,
            peer_rank=1,  # draft is rank 1
            num_speculative_tokens=spec_config.num_speculative_tokens,
            max_batch_size=vllm_config.scheduler_config.max_num_seqs,
            vocab_size=spec_config.draft_model_config.get_vocab_size(),
            device=device,
            dtype=vllm_config.model_config.dtype,
        )

        self._target_interface = DisaggDraftTargetInterface(
            communicator=communicator,
            num_speculative_tokens=spec_config.num_speculative_tokens,
            vocab_size=spec_config.draft_model_config.get_vocab_size(),
            device=device,
            dtype=vllm_config.model_config.dtype,
        )
        return self._target_interface

    @property
    def target_interface(self):
        return self._target_interface

    def shutdown(self) -> None:
        """Shutdown the draft worker process."""
        # Close death pipe first — this triggers os._exit in the
        # draft worker's death monitor thread, which is the most
        # reliable way to kill a process stuck in NCCL recv.
        if self.death_writer is not None:
            try:
                self.death_writer.close()
            except Exception:
                pass
            self.death_writer = None

        # Try graceful EXIT command (may fail if NCCL is broken).
        if self._target_interface is not None:
            try:
                self._target_interface.request_exit()
            except Exception:
                pass
            self._target_interface = None

        # Give the process a moment to exit via death monitor.
        if self.proc is not None and self.proc.is_alive():
            self.proc.join(timeout=2)

        # Force kill if still alive — NCCL recv can't be interrupted
        # by SIGTERM, so go straight to SIGKILL.
        if self.proc is not None and self.proc.is_alive():
            self.proc.kill()
            self.proc.join(timeout=3)

        if self._target_pg is not None:
            try:
                dist.destroy_process_group(self._target_pg)
            except Exception:
                pass
            self._target_pg = None


def maybe_launch_disagg_draft_worker(
    vllm_config: VllmConfig,
) -> DisaggDraftWorkerHandle | None:
    """Launch the disagg draft draft worker process if disagg draft is enabled.

    Called by the executor during initialization. If disagg draft is not enabled
    in the config, returns None. Otherwise, spawns the draft worker
    process on the designated draft GPU and returns a handle.

    The draft GPU is selected as the next GPU after the TP group.
    For example, with TP=4 using GPUs 0-3, the draft uses GPU 4.

    Args:
        vllm_config: Full vLLM configuration.

    Returns:
        DisaggDraftWorkerHandle if disagg draft is enabled, None otherwise.
    """
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_disagg_draft():
        return None

    # Determine draft GPU device
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    draft_device_id = tp_size  # Next GPU after TP group
    num_gpus = torch.cuda.device_count()

    if draft_device_id >= num_gpus:
        raise RuntimeError(
            f"Disagg draft requires an additional GPU for the draft model. "
            f"TP size is {tp_size} (using GPUs 0-{tp_size - 1}), "
            f"but only {num_gpus} GPUs are available. "
            f"Need at least {tp_size + 1} GPUs."
        )

    logger.info(
        "Disagg draft: Launching draft worker on GPU %d (TP uses GPUs 0-%d)",
        draft_device_id,
        tp_size - 1,
    )

    # Use the pre-generated NCCL init method from config (set before
    # workers were spawned so all processes share it).
    nccl_init_method = spec_config.disagg_draft_nccl_init_method
    if not nccl_init_method:
        # Fallback: generate fresh if not pre-generated
        nccl_init_method = f"tcp://{get_loopback_ip()}:{get_open_port()}"

    # Set up pipes for process lifecycle management
    context = get_mp_context()
    ready_reader, ready_writer = context.Pipe(duplex=False)
    death_reader, death_writer = context.Pipe(duplex=False)

    # Spawn the draft worker process
    proc = context.Process(
        target=launch_disagg_draft_worker,
        kwargs={
            "vllm_config": vllm_config,
            "draft_device_id": draft_device_id,
            "nccl_init_method": nccl_init_method,
            "ready_pipe": ready_writer,
            "death_pipe": death_reader,
        },
        name="DisaggDraft-Worker",
        daemon=True,
    )
    proc.start()

    # Close child ends of pipes in parent
    ready_writer.close()
    death_reader.close()

    # Wait for draft worker to be ready
    try:
        if ready_reader.poll(timeout=300):  # 5 minute timeout for model loading
            response = ready_reader.recv()
            if response.get("status") != "READY":
                raise RuntimeError(
                    f"Disagg draft draft worker failed to start: {response}"
                )
        else:
            logger.warning(
                "Disagg draft draft worker timed out during initialization (300s). "
                "The draft model may be too large or GPU %d may be busy.",
                draft_device_id,
            )
            # Don't raise — return None so the server can still start
            ready_reader.close()
            return None
    except EOFError:
        raise RuntimeError("Disagg draft draft worker process died during initialization")
    finally:
        ready_reader.close()

    logger.info("Disagg draft draft worker is ready on GPU %d.", draft_device_id)

    return DisaggDraftWorkerHandle(
        proc=proc,
        death_writer=death_writer,
        nccl_init_method=nccl_init_method,
    )
