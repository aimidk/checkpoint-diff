"""Tests for checkpoint_diff.topology."""
import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff, CheckpointDiff
from checkpoint_diff.topology import (
    TopologyRow,
    build_topology,
    format_topology,
)


def _td(
    a: np.ndarray | None,
    b: np.ndarray | None,
    status: str = "changed",
) -> TensorDiff:
    return TensorDiff(array_a=a, array_b=b, status=status)


def _make_diff(**kwargs: TensorDiff) -> CheckpointDiff:
    return dict(kwargs)


# ---------------------------------------------------------------------------
# build_topology
# ---------------------------------------------------------------------------

def test_build_topology_added_key():
    diff = _make_diff(w=_td(None, np.zeros((3, 4)), status="added"))
    rows = build_topology(diff)
    assert len(rows) == 1
    r = rows[0]
    assert r.key == "w"
    assert r.shape_a is None
    assert r.shape_b == (3, 4)
    assert r.status == "added"


def test_build_topology_removed_key():
    diff = _make_diff(b=_td(np.ones((2,)), None, status="removed"))
    rows = build_topology(diff)
    assert rows[0].shape_b is None
    assert rows[0].status == "removed"


def test_build_topology_shape_changed_flag():
    diff = _make_diff(
        w=_td(np.zeros((4, 4)), np.zeros((4, 8)), status="changed")
    )
    rows = build_topology(diff)
    assert rows[0].shape_changed is True
    assert rows[0].dtype_changed is False


def test_build_topology_dtype_changed_flag():
    a = np.zeros((3,), dtype=np.float32)
    b = np.zeros((3,), dtype=np.float64)
    diff = _make_diff(w=_td(a, b, status="changed"))
    rows = build_topology(diff)
    assert rows[0].dtype_changed is True
    assert rows[0].dtype_a == "float32"
    assert rows[0].dtype_b == "float64"


def test_build_topology_unchanged_no_flags():
    a = np.zeros((5, 5), dtype=np.float32)
    b = np.zeros((5, 5), dtype=np.float32)
    diff = _make_diff(w=_td(a, b, status="unchanged"))
    rows = build_topology(diff)
    assert rows[0].shape_changed is False
    assert rows[0].dtype_changed is False


def test_build_topology_sorted_by_key():
    diff = _make_diff(
        z=_td(np.zeros((1,)), np.zeros((1,)), status="unchanged"),
        a=_td(np.zeros((2,)), np.zeros((2,)), status="unchanged"),
    )
    rows = build_topology(diff)
    assert [r.key for r in rows] == ["a", "z"]


# ---------------------------------------------------------------------------
# format_topology
# ---------------------------------------------------------------------------

def test_format_topology_empty_returns_message():
    assert format_topology([]) == "(no tensors)"


def test_format_topology_all_identical_hides_rows():
    a = np.zeros((3,), dtype=np.float32)
    b = np.zeros((3,), dtype=np.float32)
    diff = _make_diff(w=_td(a, b, status="unchanged"))
    rows = build_topology(diff)
    result = format_topology(rows, show_unchanged=False)
    assert "identical" in result.lower()


def test_format_topology_contains_key_name():
    diff = _make_diff(encoder=_td(np.zeros((4, 4)), np.zeros((4, 8)), status="changed"))
    rows = build_topology(diff)
    result = format_topology(rows)
    assert "encoder" in result


def test_format_topology_show_unchanged_includes_all():
    a = np.zeros((3,), dtype=np.float32)
    diff = _make_diff(w=_td(a, a.copy(), status="unchanged"))
    rows = build_topology(diff)
    result = format_topology(rows, show_unchanged=True)
    assert "w" in result
