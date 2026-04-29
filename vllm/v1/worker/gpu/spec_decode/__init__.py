# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.config import VllmConfig


def init_speculator(vllm_config: VllmConfig, device: torch.device):
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.use_eagle() and not speculative_config.use_disagg():
        from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

        return EagleSpeculator(vllm_config, device)
    if speculative_config.use_disagg():
        from vllm.v1.worker.gpu.spec_decode.disagg_draft.speculator import DisaggSpeculatorProxy

        proxy = DisaggSpeculatorProxy(vllm_config, device)

        if speculative_config.uses_nm_disagg:
            # N:M mode: validate connectivity, create connectors and
            # router, and inject into the proxy.
            from vllm.v1.spec_decode.draft_connector import (
                ZmqNcclDraftConnector,
                validate_draft_server_connectivity,
            )
            from vllm.v1.spec_decode.draft_router import DraftRouter

            addresses = speculative_config.disagg_draft_addresses
            validate_draft_server_connectivity(addresses)

            # Create one ZmqNcclDraftConnector per draft server.
            # NCCL process groups are NOT available yet — the connectors
            # will use a placeholder PG and the actual tensor transport
            # will be wired when the first speculation call happens.
            # For now, ZMQ metadata channel is sufficient for the
            # handshake / healthcheck flow.
            import uuid

            verify_server_id = f"vs-{uuid.uuid4().hex[:8]}"
            timeout_ms = speculative_config.disagg_draft_timeout_ms

            connectors = []
            for addr in addresses:
                # Create connector with a dummy process group.
                # The NCCL PG will be replaced once torch.distributed
                # is initialised on the worker.
                connector = ZmqNcclDraftConnector(
                    address=addr,
                    verify_server_id=verify_server_id,
                    process_group=None,  # type: ignore[arg-type]
                    peer_rank=0,
                    device=device,
                    timeout_ms=timeout_ms,
                )
                connectors.append(connector)

            router = DraftRouter(
                connectors=connectors,
                draft_server_addresses=addresses,
                policy=speculative_config.disagg_draft_routing_policy,
            )
            proxy.set_router(router)

        return proxy
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
