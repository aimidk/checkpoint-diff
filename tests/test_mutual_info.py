"""Tests for checkpoint_diff.mutual_info."""
from __future__ import annotations

import math

import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.mutual_info import (
    MutualInfoRow,
    _histogram_entropy,
    _joint_mi,
    compute_mutual_info,
    format_mutual_info,
)


def _td(
    a: np.ndarray | None,
    b: np.ndarray | None,
    status: str = "changed",
) -> TensorDiff:
    return TensorDiff(
        status=status,
        array_a=a,
        array_b=b,
        shape_a=None if a is None else a.shape,
        shape_b=None if b is None else b.shape,
        mean_a=float(np.mean(a)) if a is not None else None,
        mean_b=float(np.mean(b)) if b is not None else None,
        std_a=float(np.std(a)) if a is not None else None,
        std_b=float(np.std(b)) if b is not None else None,
    )


def _make_diff(tensors: dict) -> CheckpointDiff:
    return CheckpointDiff(tensors=tensors)


# --- _histogram_entropy ---

def test_histogram_entropy_uniform_is_positive():
    arr = np.linspace(0, 1, 1000)
    h = _histogram_entropy(arr)
    assert h > 0


def test_histogram_entropy_constant_is_zero():
    arr = np.ones(100)
    h = _histogram_entropy(arr)
    assert h == pytest.approx(0.0, abs=1e-9)


def test_histogram_entropy_none_returns_nan():
    assert math.isnan(_histogram_entropy(None))


def test_histogram_entropy_empty_returns_nan():
    assert math.isnan(_histogram_entropy(np.array([])))


# --- _joint_mi ---

def test_joint_mi_identical_arrays_positive():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(500)
    mi = _joint_mi(arr, arr)
    assert mi > 0


def test_joint_mi_none_returns_nan():
    arr = np.ones(10)
    assert math.isnan(_joint_mi(None, arr))
    assert math.isnan(_joint_mi(arr, None))


def test_joint_mi_empty_returns_nan():
    assert math.isnan(_joint_mi(np.array([]), np.array([])))


# --- compute_mutual_info ---

def test_compute_mutual_info_skips_missing_arrays():
    diff = _make_diff({
        "w": _td(None, np.ones(10), status="added"),
    })
    rows = compute_mutual_info(diff)
    assert rows == []


def test_compute_mutual_info_returns_row_for_changed_key():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(200)
    b = a + rng.standard_normal(200) * 0.1
    diff = _make_diff({"layer.weight": _td(a, b)})
    rows = compute_mutual_info(diff)
    assert len(rows) == 1
    assert rows[0].key == "layer.weight"
    assert not math.isnan(rows[0].mi_bits)


def test_compute_mutual_info_top_n_limits_output():
    rng = np.random.default_rng(2)
    tensors = {f"k{i}": _td(rng.standard_normal(100), rng.standard_normal(100)) for i in range(5)}
    diff = _make_diff(tensors)
    rows = compute_mutual_info(diff, top_n=3)
    assert len(rows) == 3


def test_compute_mutual_info_sorted_descending():
    rng = np.random.default_rng(3)
    tensors = {f"k{i}": _td(rng.standard_normal(100), rng.standard_normal(100)) for i in range(4)}
    diff = _make_diff(tensors)
    rows = compute_mutual_info(diff)
    mis = [r.mi_bits for r in rows if not math.isnan(r.mi_bits)]
    assert mis == sorted(mis, reverse=True)


def test_normalized_mi_between_zero_and_one():
    rng = np.random.default_rng(4)
    a = rng.standard_normal(300)
    b = a * 0.9 + rng.standard_normal(300) * 0.1
    diff = _make_diff({"x": _td(a, b)})
    rows = compute_mutual_info(diff)
    nmi = rows[0].normalized_mi
    assert 0.0 <= nmi <= 1.0


# --- format_mutual_info ---

def test_format_mutual_info_contains_key():
    rng = np.random.default_rng(5)
    a = rng.standard_normal(100)
    diff = _make_diff({"encoder.weight": _td(a, a)})
    rows = compute_mutual_info(diff)
    out = format_mutual_info(rows)
    assert "encoder.weight" in out


def test_format_mutual_info_empty_returns_message():
    out = format_mutual_info([])
    assert "No mutual information" in out
