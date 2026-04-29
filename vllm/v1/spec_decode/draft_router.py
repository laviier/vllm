# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""N:M Draft Router for disaggregated speculative decoding.

The ``DraftRouter`` assigns incoming requests to available Draft_Servers
using a configurable load-balancing policy (currently round-robin).
Each Verify_Server maintains one ``DraftRouter`` that tracks which
``DraftConnector`` is responsible for each active request.
"""

from __future__ import annotations

import logging

from vllm.v1.spec_decode.draft_connector import DraftConnector

logger = logging.getLogger(__name__)


class DraftRouter:
    """Assigns requests to draft servers for N:M topology.

    Args:
        connectors: Pre-built ``DraftConnector`` instances, one per
            Draft_Server.  Connector creation (NCCL process groups,
            ZMQ sockets, etc.) is handled externally by the speculator
            proxy.
        draft_server_addresses: Addresses of the Draft_Servers, kept
            for logging / diagnostics.  Order must match *connectors*.
        policy: Load-balancing policy.  Currently only ``"round_robin"``
            is supported.
    """

    def __init__(
        self,
        connectors: list[DraftConnector],
        draft_server_addresses: list[str] | None = None,
        policy: str = "round_robin",
    ) -> None:
        if not connectors:
            raise ValueError("DraftRouter requires at least one connector")
        if policy != "round_robin":
            raise ValueError(
                f"Unsupported routing policy: {policy!r}. "
                "Only 'round_robin' is currently supported."
            )

        self.connectors = connectors
        self.draft_server_addresses = draft_server_addresses or [
            f"server-{i}" for i in range(len(connectors))
        ]
        self.policy = policy

        # request_id → server index
        self.assignment: dict[str, int] = {}

        # Per-server availability tracking
        self._available: list[bool] = [True] * len(connectors)

        # Round-robin counter
        self._next_index: int = 0

        logger.info(
            "DraftRouter initialised with %d server(s), policy=%s",
            len(connectors),
            policy,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign(self, request_id: str) -> DraftConnector:
        """Assign *request_id* to a draft server and return its connector.

        Uses round-robin across available servers.  Raises
        ``RuntimeError`` if no servers are available.
        """
        if request_id in self.assignment:
            idx = self.assignment[request_id]
            logger.debug(
                "Request %s already assigned to server %d", request_id, idx
            )
            return self.connectors[idx]

        idx = self._pick_next_available()
        self.assignment[request_id] = idx
        logger.debug("Assigned request %s to server %d", request_id, idx)
        return self.connectors[idx]

    def release(self, request_id: str) -> None:
        """Release the assignment for *request_id*."""
        idx = self.assignment.pop(request_id, None)
        if idx is not None:
            logger.debug(
                "Released request %s from server %d", request_id, idx
            )
        else:
            logger.debug(
                "Release called for unknown request %s (no-op)", request_id
            )

    def get_connector(self, request_id: str) -> DraftConnector:
        """Return the connector for an already-assigned *request_id*.

        Raises ``KeyError`` if the request has not been assigned.
        """
        idx = self.assignment[request_id]
        return self.connectors[idx]

    def handle_server_failure(self, server_index: int) -> None:
        """Mark *server_index* as unavailable and reassign its requests.

        All requests currently assigned to the failed server are
        reassigned to other available servers via round-robin.
        """
        if server_index < 0 or server_index >= len(self.connectors):
            raise IndexError(
                f"server_index {server_index} out of range "
                f"[0, {len(self.connectors)})"
            )

        self._available[server_index] = False
        addr = self.draft_server_addresses[server_index]
        logger.warning(
            "Draft server %d (%s) marked unavailable", server_index, addr
        )

        # Collect requests that need reassignment
        affected = [
            rid
            for rid, idx in self.assignment.items()
            if idx == server_index
        ]

        if not affected:
            return

        for rid in affected:
            try:
                new_idx = self._pick_next_available()
            except RuntimeError:
                # No servers available — remove assignment so the caller
                # can handle graceful degradation.
                del self.assignment[rid]
                logger.error(
                    "No available servers to reassign request %s", rid
                )
                continue

            self.assignment[rid] = new_idx
            logger.info(
                "Reassigned request %s from server %d to server %d",
                rid,
                server_index,
                new_idx,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def num_available_servers(self) -> int:
        """Number of currently available draft servers."""
        return sum(self._available)

    def mark_server_available(self, server_index: int) -> None:
        """Re-enable a previously failed server (e.g. after reconnect)."""
        if server_index < 0 or server_index >= len(self.connectors):
            raise IndexError(
                f"server_index {server_index} out of range "
                f"[0, {len(self.connectors)})"
            )
        self._available[server_index] = True
        addr = self.draft_server_addresses[server_index]
        logger.info(
            "Draft server %d (%s) marked available again",
            server_index,
            addr,
        )

    def _pick_next_available(self) -> int:
        """Return the next available server index (round-robin).

        Raises ``RuntimeError`` when no servers are available.
        """
        n = len(self.connectors)
        for _ in range(n):
            idx = self._next_index % n
            self._next_index += 1
            if self._available[idx]:
                return idx
        raise RuntimeError("No available draft servers")
