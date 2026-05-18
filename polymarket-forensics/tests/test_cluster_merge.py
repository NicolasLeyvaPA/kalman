"""Tests for the cluster merge logic.

We don't mock the database; we just test the pure functions.
"""

from __future__ import annotations

from enums import ClusterType
from services.cluster_detector import CandidateCluster, _merge_overlapping


def test_no_overlap_returns_same_count():
    a = CandidateCluster(("0xa1", "0xa2"), ClusterType.FUNDING_LINKED, "ev1")
    b = CandidateCluster(("0xb1", "0xb2"), ClusterType.TEMPORAL, "ev2")
    out = _merge_overlapping([a, b])
    assert len(out) == 2


def test_overlap_merges():
    a = CandidateCluster(("0xa1", "0xa2"), ClusterType.FUNDING_LINKED, "ev1")
    b = CandidateCluster(("0xa2", "0xa3"), ClusterType.TEMPORAL, "ev2")
    out = _merge_overlapping([a, b])
    assert len(out) == 1
    assert set(out[0].wallets) == {"0xa1", "0xa2", "0xa3"}
    assert out[0].cluster_type == ClusterType.MIXED


def test_single_type_preserved():
    a = CandidateCluster(("0xa1", "0xa2"), ClusterType.FUNDING_LINKED, "ev1")
    b = CandidateCluster(("0xa2", "0xa3"), ClusterType.FUNDING_LINKED, "ev2")
    out = _merge_overlapping([a, b])
    assert out[0].cluster_type == ClusterType.FUNDING_LINKED


def test_empty_input():
    assert _merge_overlapping([]) == []


def test_three_way_chain():
    """Transitive merge: a-b, b-c, c-d → one cluster."""
    a = CandidateCluster(("0xa", "0xb"), ClusterType.FUNDING_LINKED, "e1")
    b = CandidateCluster(("0xb", "0xc"), ClusterType.FUNDING_LINKED, "e2")
    c = CandidateCluster(("0xc", "0xd"), ClusterType.FUNDING_LINKED, "e3")
    out = _merge_overlapping([a, b, c])
    assert len(out) == 1
    assert set(out[0].wallets) == {"0xa", "0xb", "0xc", "0xd"}
