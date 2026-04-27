"""Tests for checkpoint_diff.magnitude."""
from __future__ import annotations

import math
import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff, CheckpointDiff
from checkpoint_diff.magnitude import (
    MagnitudeRow,
    _l1,
    _l2,
    _rel_change,
    compute_magnitude,
    format_magnitude,
)


def _td(a, b, status="changed") -> TensorDiff:
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    return TensorDiff(
        a=a_arr, b=b_arr, status=status,
        mean_a=float(np.mean(a_arr)), mean_b=float(np.mean(b_arr)),
        std_a=float(np.std(a_arr)), std_b=float(np.std(b_arr)),
        shape_a=a_arr.shape, shape_b=b_arr.shape,
    )


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(kwargs)


# --- unit helpers ---

def test_l1_basic():
    assert _l1(np.array([-1.0, 2.0, -3.0])) == pytest.approx(6.0)


def test_l2_basic():
    assert _l2(np.array([3.0, 4.0])) == pytest.approx(5.0)


def test_rel_change_positive():
    assert _rel_change(2.0, 3.0) == pytest.approx(0.5)


def test_rel_change_zero_denominator_returns_none():
    assert _rel_change(0.0, 5.0) is None


# --- compute_magnitude ---

def test_compute_magnitude_returns_row_per_key():
    diff = _make_diff(
        w1=_td([1, 2], [2, 3]),
        w2=_td([0, 0], [0, 0], status="unchanged"),
    )
    rows = compute_magnitude(diff)
    assert len(rows) == 2


def test_compute_magnitude_added_key_has_zero_a_norms():
    td = TensorDiff(
        a=np.zeros(0), b=np.array([1.0, 2.0]), status="added",
        mean_a=0.0, mean_b=1.5, std_a=0.0, std_b=0.5,
        shape_a=(0,), shape_b=(2,),
    )
    diff = _make_diff(new_layer=td)
    rows = compute_magnitude(diff)
    assert rows[0].norm_l2_a == pytest.approx(0.0)
    assert rows[0].rel_change_l2 is None


def test_compute_magnitude_removed_key_has_zero_b_norms():
    td = TensorDiff(
        a=np.array([3.0, 4.0]), b=np.zeros(0), status="removed",
        mean_a=3.5, mean_b=0.0, std_a=0.5, std_b=0.0,
        shape_a=(2,), shape_b=(0,),
    )
    diff = _make_diff(old_layer=td)
    rows = compute_magnitude(diff)
    assert rows[0].norm_l2_b == pytest.approx(0.0)
    assert rows[0].rel_change_l2 is None


def test_compute_magnitude_sorted_by_abs_rel_change_descending():
    diff = _make_diff(
        small=_td([10.0, 10.0], [10.1, 10.1]),
        large=_td([1.0, 0.0], [10.0, 0.0]),
    )
    rows = compute_magnitude(diff)
    assert rows[0].key == "large"


# --- format_magnitude ---

def test_format_magnitude_contains_key():
    diff = _make_diff(layer=_td([1.0, 2.0], [3.0, 4.0]))
    rows = compute_magnitude(diff)
    out = format_magnitude(rows)
    assert "layer" in out


def test_format_magnitude_empty_returns_message():
    assert "No magnitude" in format_magnitude([])


def test_format_magnitude_top_n_limits_rows():
    diff = _make_diff(
        a=_td([1, 2], [3, 4]),
        b=_td([5, 6], [7, 8]),
        c=_td([9, 10], [11, 12]),
    )
    rows = compute_magnitude(diff)
    out = format_magnitude(rows, top_n=2)
    # header + sep + 2 data rows
    data_lines = [l for l in out.splitlines() if l and not l.startswith("-") and "key" not in l]
    assert len(data_lines) == 2
