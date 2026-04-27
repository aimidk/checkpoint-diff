"""Tests for checkpoint_diff.gradient."""
from __future__ import annotations

import math

import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff
from checkpoint_diff.gradient import (
    GradientRow,
    _l2_norm,
    _rel_change,
    compute_gradient_norms,
    format_gradient_norms,
)


def _td(a, b) -> TensorDiff:
    arr_a = np.array(a, dtype=float) if a is not None else None
    arr_b = np.array(b, dtype=float) if b is not None else None
    status = "changed" if (a is not None and b is not None) else ("added" if a is None else "removed")
    return TensorDiff(
        status=status,
        array_a=arr_a,
        array_b=arr_b,
        shape_a=None if arr_a is None else arr_a.shape,
        shape_b=None if arr_b is None else arr_b.shape,
        mean_a=None if arr_a is None else float(arr_a.mean()),
        mean_b=None if arr_b is None else float(arr_b.mean()),
        std_a=None if arr_a is None else float(arr_a.std()),
        std_b=None if arr_b is None else float(arr_b.std()),
    )


def _make_diff(entries: dict) -> dict:
    return entries


def test_l2_norm_basic():
    arr = np.array([3.0, 4.0])
    assert math.isclose(_l2_norm(arr), 5.0)


def test_l2_norm_none_returns_nan():
    assert math.isnan(_l2_norm(None))


def test_rel_change_positive():
    result = _rel_change(2.0, 4.0)
    assert result > 0


def test_rel_change_nan_propagates():
    assert math.isnan(_rel_change(float("nan"), 1.0))


def test_compute_gradient_norms_returns_rows():
    diff = _make_diff({"layer.weight": _td([1.0, 0.0], [0.0, 2.0])})
    rows = compute_gradient_norms(diff)
    assert len(rows) == 1
    assert rows[0].key == "layer.weight"


def test_compute_gradient_norms_sorted_by_abs_delta():
    diff = _make_diff({
        "small": _td([1.0], [1.1]),
        "large": _td([1.0], [100.0]),
    })
    rows = compute_gradient_norms(diff)
    assert rows[0].key == "large"


def test_compute_gradient_norms_top_n_limits():
    diff = _make_diff({
        "a": _td([1.0], [2.0]),
        "b": _td([1.0], [3.0]),
        "c": _td([1.0], [4.0]),
    })
    rows = compute_gradient_norms(diff, top_n=2)
    assert len(rows) == 2


def test_compute_gradient_norms_added_key_nan_norm_a():
    diff = _make_diff({"new": _td(None, [1.0, 2.0])})
    rows = compute_gradient_norms(diff)
    assert len(rows) == 1
    assert math.isnan(rows[0].l2_norm_a)
    assert not math.isnan(rows[0].l2_norm_b)


def test_format_gradient_norms_contains_key():
    diff = _make_diff({"fc.weight": _td([1.0, 2.0], [2.0, 3.0])})
    rows = compute_gradient_norms(diff)
    output = format_gradient_norms(rows)
    assert "fc.weight" in output


def test_format_gradient_norms_empty():
    output = format_gradient_norms([])
    assert "No gradient" in output
