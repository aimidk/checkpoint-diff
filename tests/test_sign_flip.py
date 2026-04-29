"""Tests for checkpoint_diff.sign_flip."""
from __future__ import annotations

import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff, CheckpointDiff
from checkpoint_diff.sign_flip import (
    SignFlipRow,
    _flip_count,
    _pos_frac,
    compute_sign_flips,
    format_sign_flips,
)


def _td(a, b, status="changed") -> TensorDiff:
    a_arr = np.array(a, dtype=float) if a is not None else None
    b_arr = np.array(b, dtype=float) if b is not None else None
    mean_a = float(np.mean(a_arr)) if a_arr is not None else float("nan")
    mean_b = float(np.mean(b_arr)) if b_arr is not None else float("nan")
    std_a = float(np.std(a_arr)) if a_arr is not None else float("nan")
    std_b = float(np.std(b_arr)) if b_arr is not None else float("nan")
    return TensorDiff(
        status=status,
        a=a_arr,
        b=b_arr,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        shape_a=None if a_arr is None else a_arr.shape,
        shape_b=None if b_arr is None else b_arr.shape,
    )


def _make_diff(entries: dict) -> CheckpointDiff:
    return entries


def test_flip_count_all_flipped():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([-1.0, -2.0, -3.0])
    assert _flip_count(a, b) == 3


def test_flip_count_none_flipped():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert _flip_count(a, b) == 0


def test_flip_count_ignores_zeros():
    a = np.array([0.0, 1.0, -1.0])
    b = np.array([0.0, -1.0, 1.0])
    # zero in a[0] excluded; a[1]->b[1] flips, a[2]->b[2] flips
    assert _flip_count(a, b) == 2


def test_pos_frac_all_positive():
    arr = np.array([1.0, 2.0, 3.0])
    assert _pos_frac(arr) == pytest.approx(1.0)


def test_pos_frac_none_returns_nan():
    import math
    assert math.isnan(_pos_frac(None))


def test_compute_sign_flips_returns_row_for_changed_key():
    diff = _make_diff({
        "w": _td([1.0, 2.0, -1.0], [-1.0, -2.0, 1.0]),
    })
    rows = compute_sign_flips(diff)
    assert len(rows) == 1
    assert rows[0].key == "w"
    assert rows[0].flipped == 3
    assert rows[0].flip_rate == pytest.approx(1.0)


def test_compute_sign_flips_skips_removed_key():
    diff = _make_diff({
        "w": _td([1.0, 2.0], None, status="removed"),
    })
    rows = compute_sign_flips(diff)
    assert rows == []


def test_compute_sign_flips_skips_shape_mismatch():
    td = _td([1.0, 2.0], [1.0, 2.0, 3.0])
    diff = _make_diff({"w": td})
    rows = compute_sign_flips(diff)
    assert rows == []


def test_compute_sign_flips_min_rate_filter():
    diff = _make_diff({
        "high": _td([1.0, 2.0], [-1.0, -2.0]),   # 100% flip
        "low": _td([1.0, -2.0], [-1.0, 2.0]),     # 100% flip too; use partial
        "none": _td([1.0, 2.0], [3.0, 4.0]),      # 0% flip
    })
    rows = compute_sign_flips(diff, min_flip_rate=0.5)
    keys = {r.key for r in rows}
    assert "none" not in keys


def test_format_sign_flips_empty():
    assert format_sign_flips([]) == "No sign flips detected."


def test_format_sign_flips_contains_key():
    diff = _make_diff({"layer.weight": _td([1.0, 2.0], [-1.0, -2.0])})
    rows = compute_sign_flips(diff)
    out = format_sign_flips(rows)
    assert "layer.weight" in out


def test_format_sign_flips_top_n_limits_rows():
    diff = _make_diff({
        "a": _td([1.0], [-1.0]),
        "b": _td([1.0], [-1.0]),
        "c": _td([1.0], [-1.0]),
    })
    rows = compute_sign_flips(diff)
    out = format_sign_flips(rows, top_n=2)
    # header + sep + 2 data rows
    data_lines = [l for l in out.splitlines() if l and not l.startswith("-") and "Key" not in l]
    assert len(data_lines) == 2
