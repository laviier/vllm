# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Entrypoint for launching a standalone Draft Server.

Usage::

    vllm serve <draft_model> --draft-server --draft-server-port 50051 \
        --speculative-config '{"num_speculative_tokens": 5, "method": "eagle"}'

The module parses CLI args, builds a ``VllmConfig``, and starts the
``DraftServer`` from ``vllm.v1.spec_decode.draft_server``.
"""

import argparse
import asyncio
import signal

import uvloop

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext

logger = init_logger(__name__)


def _init_distributed_for_draft_server(vllm_config) -> None:
    """Initialize torch.distributed and vLLM parallel state for TP=1.

    Must be called inside a ``set_current_vllm_config`` context so that
    ``initialize_model_parallel`` can read the current config.
    """
    from vllm.distributed.parallel_state import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
        model_parallel_is_initialized,
    )

    if not model_parallel_is_initialized():
        from vllm.utils.network_utils import get_open_port
        init_port = get_open_port()
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{init_port}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    logger.info("Draft server distributed environment initialized (TP=1).")


def run_draft_server(args: argparse.Namespace) -> None:
    """Parse engine args, create VllmConfig, and run the DraftServer."""
    if hasattr(args, "model_tag") and args.model_tag is not None:
        args.model = args.model_tag

    engine_args = AsyncEngineArgs.from_cli_args(args)

    if engine_args.speculative_config is not None:
        if "model" not in engine_args.speculative_config:
            engine_args.speculative_config["model"] = engine_args.model

    vllm_config = engine_args.create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER,
    )

    if vllm_config.speculative_config is None:
        raise ValueError(
            "--draft-server requires --speculative-config to be set "
            "with at least the draft model and num_speculative_tokens."
        )

    port = getattr(args, "draft_server_port", 50051)
    device_id = getattr(args, "draft_server_device", None)
    bind_address = f"tcp://*:{port}"

    # Set the CUDA device for the draft server process.
    # This allows running without CUDA_VISIBLE_DEVICES isolation
    # so NCCL can see all GPUs for P2P transfers.
    if device_id is not None:
        import torch
        torch.cuda.set_device(device_id)
        logger.info("Draft server using GPU %d", device_id)

    # Expose Prometheus metrics over HTTP so operators can scrape
    # cache hit rate, batch size, generation latency, etc. Uses the
    # ZMQ port + 1000 so 50051 → 51051, 50052 → 51052. Override with
    # DRAFT_METRICS_PORT env var if that collides.
    import os
    import prometheus_client
    metrics_port = int(
        os.environ.get("DRAFT_METRICS_PORT", port + 1000)
    )
    try:
        prometheus_client.start_http_server(metrics_port)
        logger.info(
            "Draft Server metrics on http://0.0.0.0:%d/metrics",
            metrics_port,
        )
    except OSError as e:
        logger.warning(
            "Could not start metrics HTTP server on port %d: %s",
            metrics_port, e,
        )

    logger.info("Starting Draft Server on %s", bind_address)

    # Everything that touches model parallel state or model loading
    # must run inside the vllm config context.
    from vllm.config.vllm import set_current_vllm_config

    with set_current_vllm_config(vllm_config):
        _init_distributed_for_draft_server(vllm_config)

        from vllm.v1.spec_decode.draft_server import DraftServer

        server = DraftServer(vllm_config, bind_address=bind_address)

        logger.info("Loading draft model...")
        server.load_model()
        logger.info("Draft model loaded, starting server loop.")

    serve_task: asyncio.Task | None = None

    def _signal_handler(signum: int, frame: object) -> None:
        logger.info("Received signal %d, shutting down Draft Server…", signum)
        if serve_task is not None and not serve_task.done():
            # Schedule cancellation from within the event loop thread
            serve_task.get_loop().call_soon_threadsafe(serve_task.cancel)

    def _profile_handler(signum: int, frame: object) -> None:
        # SIGUSR1 = start, SIGUSR2 = stop. Set
        # VLLM_DRAFT_TORCH_PROFILER_DIR before launching the server.
        if serve_task is None:
            logger.warning(
                "Received profile signal %d but server not running.", signum,
            )
            return
        loop = serve_task.get_loop()
        if signum == signal.SIGUSR1:
            loop.call_soon_threadsafe(server.start_profile)
        elif signum == signal.SIGUSR2:
            loop.call_soon_threadsafe(server.stop_profile)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGUSR1, _profile_handler)
    signal.signal(signal.SIGUSR2, _profile_handler)

    async def _run() -> None:
        nonlocal serve_task
        serve_task = asyncio.create_task(server.serve())
        try:
            await serve_task
        except asyncio.CancelledError:
            pass
        finally:
            await server.shutdown()

    uvloop.run(_run())
