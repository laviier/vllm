# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixin modules for ``DraftServer`` to keep concerns in separate files.

The mixins assume the consumer (``DraftServer``) initialises:

    self._request_state, self._verify_servers,
    self._next_internal_seq_id, self._free_internal_seq_ids,
    self._ext_to_int_seq, self._int_to_ext_seq,
    self._socket, self._current_tensor_frames, self._current_tensor_idx,
    self._buffer_counter, self.metrics, self.device, self.dtype, …

plus the per-mixin attributes documented in each file.
"""

from vllm.v1.spec_decode.draft_server_mixins.cache_build import (
    DraftServerCacheBuildMixin,
)
from vllm.v1.spec_decode.draft_server_mixins.seq_id_mapping import (
    DraftServerSeqIdMixin,
)
from vllm.v1.spec_decode.draft_server_mixins.speculate_handler import (
    DraftServerSpeculateMixin,
)
from vllm.v1.spec_decode.draft_server_mixins.transport import (
    DraftServerTransportMixin,
)

__all__ = [
    "DraftServerCacheBuildMixin",
    "DraftServerSeqIdMixin",
    "DraftServerSpeculateMixin",
    "DraftServerTransportMixin",
]
