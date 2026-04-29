"""Tests for checkpoint_diff.activation."""
from __future__ import annotations

import math
import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff, CheckpointDiff
from checkpoint_diff.activation import (
    ActivationRow,
    _fractions,
    compute_activations,
    format_activations,
)


def _td(a, b, status="changed"):
    return TensorDiff(
        status=status,
        array_a=np.array(a, dtype=float) if a is not None else None,
        array_b=np.array(b, dtype=float) if b is not None else None,
        shape_a=None,
        shape_b=None,
        mean_a=None, mean_b=None,
        std_a=None, std_b=None,
    )


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(tensors=kwargs)


# --- _fractions ---

def test_fractions_basic():
    pos, neg, zero = _fractions(np.array([1.0, -1.0, 0.0, 2.0]))
    assert pos == pytest.approx(0.5)
    assert neg == pytest.approx(0.25)
    assert zero == pytest.approx(0.25)


def test_fractions_all_positive():
    pos, neg, zero = _fractions(np.array([1.0, 2.0, 3.0]))
    assert pos == pytest.approx(1.0)
    assert neg == pytest.approx(0.0)
    assert zero == pytest.approx(0.0)


def test_fractions_none_returns_nan():
    pos, neg, zero = _fractions(None)
    assert math.isnan(pos) and math.isnan(neg) and math.isnan(zero)


def test_fractions_empty_returns_nan():
    pos, neg, zero = _fractions(np.array([]))
    assert math.isnan(pos)


# --- compute_activations ---

def test_compute_activations_returns_row_per_changed_key():
    diff = _make_diff(
        w1=_td([1.0, -1.0, 0.0], [1.0, 1.0, 0.0]),
        w2=_td([0.0, 0.0], [0.0, 0.0]),
    )
    rows = compute_activations(diff)
    assert len(rows) == 2


def test_compute_activations_skips_removed():
    diff = _make_diff(
        w1=_td(None, [1.0, 2.0], status="added"),
        w2=_td([1.0], None, status="removed"),
    )
    rows = compute_activations(diff)
    keys = [r.key for r in rows]
    assert "w2" not in keys
    assert "w1" in keys


def test_compute_activations_skips_unchanged_by_default():
    diff = _make_diff(
        w1=_td([1.0], [1.0], status="unchanged"),
    )
    rows = compute_activations(diff)
    assert rows == []


def test_compute_activations_includes_unchanged_when_flag_set():
    diff = _make_diff(
        w1=_td([1.0], [1.0], status="unchanged"),
    )
    rows = compute_activations(diff, include_unchanged=True)
    assert len(rows) == 1


def test_compute_activations_sorted_by_abs_pos_delta():
    diff = _make_diff(
        small=_td([1.0, -1.0], [1.0, -0.9]),
        large=_td([-1.0, -1.0], [1.0, 1.0]),
    )
    rows = compute_activations(diff)
    assert rows[0].key == "large"


def test_compute_activations_top_n():
    diff = _make_diff(
        a=_td([1.0], [0.0]),
        b=_td([0.0], [1.0]),
        c=_td([1.0, 1.0], [0.0, 0.0]),
    )
    rows = compute_activations(diff, top_n=2)
    assert len(rows) == 2


# --- format_activations ---

def test_format_activations_empty():
    assert format_activations([]) == "No activation data."


def test_format_activations_contains_key():
    diff = _make_diff(layer=_td([1.0, -1.0, 0.0], [1.0, 1.0, 0.0]))
    rows = compute_activations(diff)
    out = format_activations(rows)
    assert "layer" in out
    assert "pos_a" in out
