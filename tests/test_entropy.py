"""Tests for checkpoint_diff.entropy."""
import math

import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.entropy import (
    EntropyRow,
    _histogram_entropy,
    compute_entropy,
    format_entropy,
)


def _td(a, b, status="changed"):
    return TensorDiff(
        tensor_a=np.array(a, dtype=float) if a is not None else None,
        tensor_b=np.array(b, dtype=float) if b is not None else None,
        status=status,
    )


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(kwargs)


# ---------------------------------------------------------------------------
# _histogram_entropy
# ---------------------------------------------------------------------------

def test_histogram_entropy_uniform_is_max():
    arr = np.linspace(0, 1, 1000)
    h = _histogram_entropy(arr, bins=10)
    assert h == pytest.approx(math.log2(10), rel=0.05)


def test_histogram_entropy_constant_array_is_zero():
    arr = np.ones(100)
    h = _histogram_entropy(arr, bins=10)
    assert h == pytest.approx(0.0, abs=1e-9)


def test_histogram_entropy_empty_array_is_nan():
    h = _histogram_entropy(np.array([]))
    assert math.isnan(h)


# ---------------------------------------------------------------------------
# compute_entropy
# ---------------------------------------------------------------------------

def test_compute_entropy_returns_row_for_changed_key():
    diff = _make_diff(w=_td([1, 2, 3], [4, 5, 6]))
    rows = compute_entropy(diff)
    assert len(rows) == 1
    assert rows[0].key == "w"


def test_compute_entropy_skips_removed_key():
    diff = _make_diff(
        w=_td([1, 2, 3], None, status="removed"),
        v=_td([1, 2, 3], [4, 5, 6]),
    )
    keys = [r.key for r in compute_entropy(diff)]
    assert "w" not in keys
    assert "v" in keys


def test_compute_entropy_excludes_unchanged_by_default():
    diff = _make_diff(
        w=_td([1, 2, 3], [1, 2, 3], status="unchanged"),
        v=_td([1, 2, 3], [4, 5, 6]),
    )
    keys = [r.key for r in compute_entropy(diff)]
    assert "w" not in keys


def test_compute_entropy_includes_unchanged_when_flag_set():
    diff = _make_diff(
        w=_td([1, 2, 3], [1, 2, 3], status="unchanged"),
    )
    rows = compute_entropy(diff, include_unchanged=True)
    assert any(r.key == "w" for r in rows)


def test_compute_entropy_added_key_has_nan_entropy_a():
    diff = _make_diff(w=_td(None, [1, 2, 3, 4], status="added"))
    rows = compute_entropy(diff)
    assert len(rows) == 1
    assert math.isnan(rows[0].entropy_a)
    assert not math.isnan(rows[0].entropy_b)
    assert math.isnan(rows[0].delta)


def test_compute_entropy_sorted_by_abs_delta_descending():
    diff = _make_diff(
        small=_td(np.ones(50), np.ones(50) * 1.001),
        large=_td(np.linspace(0, 1, 200), np.random.default_rng(0).uniform(0, 10, 200)),
    )
    rows = compute_entropy(diff)
    assert rows[0].key == "large"


# ---------------------------------------------------------------------------
# format_entropy
# ---------------------------------------------------------------------------

def test_format_entropy_empty_returns_message():
    assert format_entropy([]) == "No entropy data."


def test_format_entropy_contains_key():
    diff = _make_diff(bias=_td([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]))
    rows = compute_entropy(diff)
    out = format_entropy(rows)
    assert "bias" in out


def test_format_entropy_top_n_limits_rows():
    diff = _make_diff(
        a=_td([1, 2], [3, 4]),
        b=_td([5, 6], [7, 8]),
        c=_td([9, 10], [11, 12]),
    )
    rows = compute_entropy(diff)
    out = format_entropy(rows, top_n=2)
    assert out.count("\n") < len(rows) + 2  # header + sep + at most 2 data rows
