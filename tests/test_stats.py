"""Tests for checkpoint_diff.stats."""

from __future__ import annotations

import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.stats import TensorStats, compute_stats, format_stats


def _td(
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    status: str = "changed",
) -> TensorDiff:
    return TensorDiff(
        tensor_a=a,
        tensor_b=b,
        status=status,
        shape_a=a.shape if a is not None else None,
        shape_b=b.shape if b is not None else None,
        mean_a=float(np.mean(a)) if a is not None else None,
        mean_b=float(np.mean(b)) if b is not None else None,
        std_a=float(np.std(a)) if a is not None else None,
        std_b=float(np.std(b)) if b is not None else None,
    )


def _make_diff(**kwargs: TensorDiff) -> CheckpointDiff:
    return CheckpointDiff(kwargs)


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------

def test_compute_stats_returns_entry_for_changed_key():
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    diff = _make_diff(w=_td(a=arr * 0.9, b=arr, status="changed"))
    result = compute_stats(diff)
    assert "w" in result


def test_compute_stats_skips_removed_key():
    arr = np.array([1.0, 2.0])
    diff = _make_diff(old_w=_td(a=arr, b=None, status="removed"))
    result = compute_stats(diff)
    assert "old_w" not in result


def test_compute_stats_includes_added_key():
    arr = np.array([0.5, 1.5])
    diff = _make_diff(new_w=_td(a=None, b=arr, status="added"))
    result = compute_stats(diff)
    assert "new_w" in result


def test_stats_values_are_correct():
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    diff = _make_diff(layer=_td(a=arr, b=arr, status="unchanged"))
    result = compute_stats(diff)
    s: TensorStats = result["layer"]
    assert s.min == pytest.approx(1.0)
    assert s.max == pytest.approx(4.0)
    assert s.mean == pytest.approx(2.5)
    assert s.num_elements == 4
    assert s.l2_norm == pytest.approx(np.linalg.norm(arr))
    assert s.median == pytest.approx(2.5)


def test_compute_stats_empty_diff():
    diff = _make_diff()
    assert compute_stats(diff) == {}


# ---------------------------------------------------------------------------
# format_stats
# ---------------------------------------------------------------------------

def test_format_stats_empty_returns_message():
    assert "No statistics" in format_stats({})


def test_format_stats_contains_key():
    arr = np.ones(8)
    diff = _make_diff(bias=_td(a=arr, b=arr, status="unchanged"))
    stats = compute_stats(diff)
    output = format_stats(stats)
    assert "bias" in output


def test_format_stats_top_n_limits_rows():
    diff = _make_diff(
        a=_td(b=np.ones(4), status="added"),
        b=_td(b=np.ones(8) * 2, status="added"),
        c=_td(b=np.ones(2) * 0.1, status="added"),
    )
    stats = compute_stats(diff)
    output = format_stats(stats, top_n=2)
    # header + sep + 2 data rows
    data_lines = [l for l in output.splitlines() if l and not l.startswith("-") and not l.startswith("Key")]
    assert len(data_lines) == 2
