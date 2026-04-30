"""Tests for checkpoint_diff.effective_rank."""
from __future__ import annotations

import math

import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.effective_rank import (
    EffectiveRankRow,
    _effective_rank,
    compute_effective_rank,
    format_effective_rank,
)


def _td(a, b, status="changed"):
    return TensorDiff(
        array_a=np.array(a, dtype=float) if a is not None else None,
        array_b=np.array(b, dtype=float) if b is not None else None,
        status=status,
    )


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(kwargs)


# --- _effective_rank ---

def test_effective_rank_none_returns_nan():
    assert math.isnan(_effective_rank(None))


def test_effective_rank_empty_returns_nan():
    assert math.isnan(_effective_rank(np.array([])))


def test_effective_rank_constant_vector_is_one():
    arr = np.ones((4,))
    result = _effective_rank(arr)
    assert abs(result - 1.0) < 1e-6


def test_effective_rank_identity_matrix_equals_size():
    n = 4
    arr = np.eye(n)
    result = _effective_rank(arr)
    assert abs(result - n) < 1e-6


def test_effective_rank_rank1_matrix_is_one():
    v = np.array([1.0, 2.0, 3.0])
    arr = np.outer(v, v)
    result = _effective_rank(arr)
    assert abs(result - 1.0) < 1e-6


# --- compute_effective_rank ---

def test_compute_effective_rank_skips_removed():
    diff = _make_diff(w=_td([1, 2], None, status="removed"))
    rows = compute_effective_rank(diff)
    assert rows == []


def test_compute_effective_rank_returns_row_for_changed():
    diff = _make_diff(w=_td([[1, 0], [0, 1]], [[2, 0], [0, 2]]))
    rows = compute_effective_rank(diff)
    assert len(rows) == 1
    assert rows[0].key == "w"


def test_compute_effective_rank_delta_is_rank_b_minus_rank_a():
    a = np.eye(4)
    b = np.ones((4, 4))
    diff = _make_diff(w=_td(a.tolist(), b.tolist()))
    rows = compute_effective_rank(diff)
    expected_delta = _effective_rank(b) - _effective_rank(a)
    assert abs(rows[0].delta - expected_delta) < 1e-6


def test_compute_effective_rank_sorted_by_abs_delta():
    a1, b1 = np.eye(4), np.ones((4, 4))  # large delta
    a2, b2 = np.eye(3), np.eye(3)        # zero delta
    diff = _make_diff(x=_td(a2.tolist(), b2.tolist()), y=_td(a1.tolist(), b1.tolist()))
    rows = compute_effective_rank(diff)
    assert rows[0].key == "y"


# --- format_effective_rank ---

def test_format_effective_rank_empty_returns_message():
    result = format_effective_rank([])
    assert "No effective rank" in result


def test_format_effective_rank_contains_key():
    rows = [EffectiveRankRow(key="layer.weight", rank_a=2.0, rank_b=3.0, delta=1.0, status="changed")]
    result = format_effective_rank(rows)
    assert "layer.weight" in result


def test_format_effective_rank_top_n_limits_rows():
    rows = [
        EffectiveRankRow(key=f"k{i}", rank_a=1.0, rank_b=2.0, delta=float(i), status="changed")
        for i in range(5)
    ]
    result = format_effective_rank(rows, top_n=2)
    assert result.count("k") == 2
