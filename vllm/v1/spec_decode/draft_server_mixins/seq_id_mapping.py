# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-request bookkeeping and external↔internal seq_id translation.

Each verify server numbers its own seq_ids independently; the draft
server assigns globally-unique internal seq_ids so KV cache and
speculation-cache state stay isolated across VSes. This mixin owns
the small bookkeeping for that mapping plus the per-request state
dict.

Expects the consumer to initialise:

    self._request_state: dict[RequestKey, dict[str, Any]]
    self._verify_servers: dict[str, set[RequestKey]]
    self._next_internal_seq_id: int
    self._free_internal_seq_ids: list[int]
    self._ext_to_int_seq: dict[tuple[str, int], int]
    self._int_to_ext_seq: dict[int, tuple[str, int]]
    self.metrics  (Prometheus collector with ``draft_active_requests``)
"""

from __future__ import annotations

import torch

# Composite key shared with draft_server.py for type readability.
RequestKey = tuple[str, int]


class DraftServerSeqIdMixin:
    """Mixin: per-request state + seq_id remap for ``DraftServer``."""

    def _make_key(self, verify_server_id: str, seq_id: int) -> RequestKey:
        """Create a composite key for per-request state."""
        return (verify_server_id, seq_id)

    def _register_request(
        self, verify_server_id: str, seq_id: int
    ) -> RequestKey:
        """Register a new request and return its composite key."""
        key = self._make_key(verify_server_id, seq_id)
        if key not in self._request_state:
            self._request_state[key] = {}
        # Track under the verify server
        if verify_server_id not in self._verify_servers:
            self._verify_servers[verify_server_id] = set()
        self._verify_servers[verify_server_id].add(key)
        # Update active request count metric.
        self.metrics.draft_active_requests.set(len(self._request_state))
        return key

    def _unregister_request(
        self, verify_server_id: str, seq_id: int
    ) -> None:
        """Remove a request's state and tracking."""
        key = self._make_key(verify_server_id, seq_id)
        self._request_state.pop(key, None)
        server_keys = self._verify_servers.get(verify_server_id)
        if server_keys is not None:
            server_keys.discard(key)
            if not server_keys:
                del self._verify_servers[verify_server_id]
        # Update active request count metric.
        self.metrics.draft_active_requests.set(len(self._request_state))

    def _alloc_internal_seq_id(self) -> int:
        """Allocate a unique internal seq_id."""
        if self._free_internal_seq_ids:
            return self._free_internal_seq_ids.pop()
        sid = self._next_internal_seq_id
        self._next_internal_seq_id += 1
        return sid

    def _map_seq_id(self, vs_id: str, ext_seq_id: int) -> int:
        """Map (verify_server_id, external_seq_id) → internal_seq_id.

        Allocates a new internal ID on first use.
        """
        key = (vs_id, ext_seq_id)
        if key not in self._ext_to_int_seq:
            internal = self._alloc_internal_seq_id()
            self._ext_to_int_seq[key] = internal
            self._int_to_ext_seq[internal] = key
        return self._ext_to_int_seq[key]

    def _unmap_seq_id(self, vs_id: str, ext_seq_id: int) -> int | None:
        """Remove mapping and recycle the internal seq_id."""
        key = (vs_id, ext_seq_id)
        internal = self._ext_to_int_seq.pop(key, None)
        if internal is not None:
            self._int_to_ext_seq.pop(internal, None)
            self._free_internal_seq_ids.append(internal)
        return internal

    def _remap_seq_ids(
        self, vs_id: str, seq_ids: torch.Tensor
    ) -> torch.Tensor:
        """Remap a tensor of external seq_ids to internal seq_ids."""
        internal_ids = []
        for ext_id in seq_ids.tolist():
            internal_ids.append(self._map_seq_id(vs_id, int(ext_id)))
        return torch.tensor(
            internal_ids, dtype=seq_ids.dtype, device=seq_ids.device
        )
