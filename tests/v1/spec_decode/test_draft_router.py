# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DraftRouter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vllm.v1.spec_decode.draft_router import DraftRouter


def _make_connectors(n: int) -> list[MagicMock]:
    """Create *n* mock DraftConnector instances."""
    return [MagicMock(name=f"connector-{i}") for i in range(n)]


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------


class TestDraftRouterInit:
    def test_requires_at_least_one_connector(self):
        with pytest.raises(ValueError, match="at least one connector"):
            DraftRouter(connectors=[])

    def test_rejects_unsupported_policy(self):
        with pytest.raises(ValueError, match="Unsupported routing policy"):
            DraftRouter(connectors=_make_connectors(1), policy="random")

    def test_default_addresses_generated(self):
        router = DraftRouter(connectors=_make_connectors(2))
        assert router.draft_server_addresses == ["server-0", "server-1"]

    def test_custom_addresses(self):
        addrs = ["tcp://a:1", "tcp://b:2"]
        router = DraftRouter(
            connectors=_make_connectors(2),
            draft_server_addresses=addrs,
        )
        assert router.draft_server_addresses == addrs

    def test_affinity_explicit_primary(self):
        conns = _make_connectors(2)
        router = DraftRouter(
            connectors=conns,
            policy="affinity",
            primary_index=1,
        )
        assert router.assign("r0") is conns[1]
        assert router.assign("r1") is conns[1]

    def test_affinity_explicit_primary_must_be_in_range(self):
        with pytest.raises(ValueError, match="primary_index 2 out of range"):
            DraftRouter(
                connectors=_make_connectors(2),
                policy="affinity",
                primary_index=2,
            )

    def test_affinity_explicit_primary_overrides_hash(self):
        conns = _make_connectors(2)
        router = DraftRouter(
            connectors=conns,
            policy="affinity",
            verify_server_id="any-id",
            primary_index=1,
        )
        assert router.assign("r0") is conns[1]

    def test_affinity_explicit_primary_fails_over(self):
        conns = _make_connectors(2)
        router = DraftRouter(
            connectors=conns,
            policy="affinity",
            primary_index=0,
        )
        assert router.assign("r0") is conns[0]
        assert router.handle_server_failure(0) == ["r0"]
        assert router.get_connector("r0") is conns[1]


# ------------------------------------------------------------------
# assign / release / get_connector
# ------------------------------------------------------------------


class TestAssignRelease:
    def test_assign_returns_connector(self):
        conns = _make_connectors(2)
        router = DraftRouter(connectors=conns)
        c = router.assign("req-1")
        assert c in conns

    def test_assign_round_robin(self):
        conns = _make_connectors(3)
        router = DraftRouter(connectors=conns)
        c0 = router.assign("r0")
        c1 = router.assign("r1")
        c2 = router.assign("r2")
        c3 = router.assign("r3")
        assert c0 is conns[0]
        assert c1 is conns[1]
        assert c2 is conns[2]
        assert c3 is conns[0]  # wraps around

    def test_assign_idempotent(self):
        conns = _make_connectors(2)
        router = DraftRouter(connectors=conns)
        c1 = router.assign("req-1")
        c2 = router.assign("req-1")
        assert c1 is c2

    def test_release_removes_assignment(self):
        router = DraftRouter(connectors=_make_connectors(1))
        router.assign("req-1")
        router.release("req-1")
        assert "req-1" not in router.assignment

    def test_release_unknown_is_noop(self):
        router = DraftRouter(connectors=_make_connectors(1))
        router.release("nonexistent")  # should not raise

    def test_get_connector_returns_assigned(self):
        conns = _make_connectors(2)
        router = DraftRouter(connectors=conns)
        router.assign("req-1")
        assert router.get_connector("req-1") in conns

    def test_get_connector_raises_for_unknown(self):
        router = DraftRouter(connectors=_make_connectors(1))
        with pytest.raises(KeyError):
            router.get_connector("unknown")


# ------------------------------------------------------------------
# handle_server_failure
# ------------------------------------------------------------------


class TestHandleServerFailure:
    def test_marks_server_unavailable(self):
        router = DraftRouter(connectors=_make_connectors(2))
        router.handle_server_failure(0)
        assert router._available[0] is False
        assert router._available[1] is True

    def test_reassigns_affected_requests(self):
        conns = _make_connectors(2)
        router = DraftRouter(connectors=conns)
        router.assign("r0")  # → server 0
        router.assign("r1")  # → server 1
        affected = router.handle_server_failure(0)
        # r0 should now be on server 1
        assert router.assignment["r0"] == 1
        assert affected == ["r0"]

    def test_failure_of_all_servers_removes_assignments(self):
        router = DraftRouter(connectors=_make_connectors(1))
        router.assign("r0")
        affected = router.handle_server_failure(0)
        # No servers left — assignment removed
        assert "r0" not in router.assignment
        assert affected == ["r0"]

    def test_failure_without_assignments_returns_empty_list(self):
        router = DraftRouter(connectors=_make_connectors(2))
        assert router.handle_server_failure(0) == []

    def test_invalid_server_index_raises(self):
        router = DraftRouter(connectors=_make_connectors(2))
        with pytest.raises(IndexError):
            router.handle_server_failure(5)
        with pytest.raises(IndexError):
            router.handle_server_failure(-1)

    def test_assign_skips_unavailable(self):
        conns = _make_connectors(3)
        router = DraftRouter(connectors=conns)
        router.handle_server_failure(0)
        c = router.assign("r0")
        # Should skip server 0
        assert c is not conns[0]

    def test_all_unavailable_raises_on_assign(self):
        router = DraftRouter(connectors=_make_connectors(2))
        router.handle_server_failure(0)
        router.handle_server_failure(1)
        with pytest.raises(RuntimeError, match="No available draft servers"):
            router.assign("r0")


# ------------------------------------------------------------------
# mark_server_available
# ------------------------------------------------------------------


class TestMarkServerAvailable:
    def test_re_enables_server(self):
        router = DraftRouter(connectors=_make_connectors(2))
        router.handle_server_failure(0)
        assert router.num_available_servers == 1
        router.mark_server_available(0)
        assert router.num_available_servers == 2
        assert router._available[0] is True

    def test_invalid_index_raises(self):
        router = DraftRouter(connectors=_make_connectors(1))
        with pytest.raises(IndexError):
            router.mark_server_available(5)
