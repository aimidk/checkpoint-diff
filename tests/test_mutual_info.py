"""Tests for checkpoint_diff.mutual_info."""
from __future__ import annotations

import math
import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff, CheckpointDiff
from checkpoint_diff.mutual_info import (
    _histogram_entropy,
    _joint_entropy,
    compute_mutual_info,
    format_mutual_info,
)


def _td(key, a, b, status="changed"):
    return TensorDiff(
        key=key,
        status=status,
        array_a=np.array(a, dtype=float) if a is not None else None,
        array_b=np.array(b, dtype=float) if b is not None else None,
        mean_a=float(np.mean(a)) if a is not None else None,
        mean_b=float(np.mean(b)) if b is not None else None,
        std_a=float(np.std(a)) if a is not None else None,
        std_b=float(np.std(b)) if b is not None else None,
        shape_a=np.array(a).shape if a is not None else None,
        shape_b=np.array(b).shape if b is not None else None,
    )


def _make_diff(*tds):
    return {td.key: td for td in tds}


def test_histogram_entropy_uniform_positive():
    arr = np.linspace(0, 1, 1000)
    h = _histogram_entropy(arr, bins=32)
    assert h > 0


def test_histogram_entropy_constant_is_zero():
    arr = np.ones(100)
    h = _histogram_entropy(arr, bins=32)
    assert h == 0.0


def test_histogram_entropy_none_returns_nan():
    h = _histogram_entropy(None, bins=32)
    assert math.isnan(h)


def test_histogram_entropy_empty_returns_nan():
    h = _histogram_entropy(np.array([]), bins=32)
    assert math.isnan(h)


def test_joint_entropy_identical_arrays():
    a = np.linspace(0, 1, 500)
    h_joint = _joint_entropy(a, a, bins=32)
    h_marg = _histogram_entropy(a, bins=32)
    # Joint entropy of identical vars should equal marginal entropy
    assert abs(h_joint - h_marg) < 0.5  # histogram approximation tolerance


def test_compute_mi_identical_tensors_high_nmi():
    a = np.random.default_rng(0).normal(size=500)
    diff = _make_diff(_td("w", a.tolist(), a.tolist()))
    rows = compute_mutual_info(diff)
    assert len(rows) == 1
    assert rows[0].normalized_mi > 0.7


def test_compute_mi_skips_missing_tensor():
    diff = _make_diff(
        _td("added", None, [1, 2, 3], status="added"),
        _td("removed", [1, 2, 3], None, status="removed"),
    )
    rows = compute_mutual_info(diff)
    assert rows == []


def test_compute_mi_top_n_limits_results():
    rng = np.random.default_rng(42)
    diff = _make_diff(
        _td("a", rng.normal(size=200).tolist(), rng.normal(size=200).tolist()),
        _td("b", rng.normal(size=200).tolist(), rng.normal(size=200).tolist()),
        _td("c", rng.normal(size=200).tolist(), rng.normal(size=200).tolist()),
    )
    rows = compute_mutual_info(diff, top_n=2)
    assert len(rows) == 2


def test_compute_mi_sorted_descending():
    rng = np.random.default_rng(7)
    base = rng.normal(size=300)
    diff = _make_diff(
        _td("high", base.tolist(), base.tolist()),           # identical → high MI
        _td("low", base.tolist(), rng.normal(size=300).tolist()),  # independent → low MI
    )
    rows = compute_mutual_info(diff)
    assert rows[0].mi_bits >= rows[1].mi_bits


def test_format_mutual_info_contains_key():
    a = np.linspace(-1, 1, 100)
    diff = _make_diff(_td("layer.weight", a.tolist(), a.tolist()))
    rows = compute_mutual_info(diff)
    out = format_mutual_info(rows)
    assert "layer.weight" in out
    assert "MI (bits)" in out


def test_format_mutual_info_empty():
    out = format_mutual_info([])
    assert "No mutual information" in out
