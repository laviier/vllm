# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.config import VllmConfig


def init_speculator(vllm_config: VllmConfig, device: torch.device):
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.use_disagg():
        import uuid

        from vllm.v1.spec_decode.draft_connector import (
            ZmqDraftConnector,
            validate_draft_server_connectivity,
        )
        from vllm.v1.spec_decode.draft_router import DraftRouter
        from vllm.v1.worker.gpu.spec_decode.disagg_draft.speculator import (
            DisaggSpeculatorProxy,
        )

        addresses = speculative_config.disagg_draft_addresses
        validate_draft_server_connectivity(addresses)

        verify_server_id = f"vs-{uuid.uuid4().hex[:8]}"
        timeout_ms = speculative_config.disagg_draft_timeout_ms
        connectors = [
            ZmqDraftConnector(
                address=addr,
                verify_server_id=verify_server_id,
                device=device,
                timeout_ms=timeout_ms,
            )
            for addr in addresses
        ]
        router = DraftRouter(
            connectors=connectors,
            draft_server_addresses=addresses,
            policy=speculative_config.disagg_draft_routing_policy,
            verify_server_id=verify_server_id,
        )
        proxy = DisaggSpeculatorProxy(vllm_config, device)
        proxy.set_router(router)
        return proxy
    if speculative_config.method == "dflash":
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
            DFlashSpeculator,
        )

        return DFlashSpeculator(vllm_config, device)
    elif speculative_config.method == "dspark":
        from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
            DSparkSpeculator,
        )

        return DSparkSpeculator(vllm_config, device)
    elif speculative_config.use_gemma4_mtp():
        from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import (
            Gemma4Speculator,
        )

        return Gemma4Speculator(vllm_config, device)
    elif speculative_config.method == "mtp":
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

        return MTPSpeculator(vllm_config, device)
    elif speculative_config.use_eagle():
        from vllm.v1.worker.gpu.spec_decode.eagle.speculator import (
            EagleSpeculator,
        )

        return EagleSpeculator(vllm_config, device)
    else:
        raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
