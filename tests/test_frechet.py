"""Tests for checkpoint_diff.frechet."""
from __future__ import annotations

import math

import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.frechet import (
    FrechetRow,
    _frechet,
    compute_frechet,
    format_frechet,
)


def _td(a, b, status="changed") -> TensorDiff:
    return TensorDiff(
        status=status,
        array_a=np.array(a, dtype=float) if a is not None else None,
        array_b=np.array(b, dtype=float) if b is not None else None,
        shape_a=None if a is None else np.array(a).shape,
        shape_b=None if b is None else np.array(b).shape,
        mean_a=None,
        mean_b=None,
        std_a=None,
        std_b=None,
    )


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(tensors=kwargs)


# --- _frechet unit tests ---

def test_frechet_identical_arrays_returns_zero():
    a = np.array([1.0, 2.0, 3.0])
    assert _frechet(a, a) == pytest.approx(0.0)


def test_frechet_none_a_returns_nan():
    assert math.isnan(_frechet(None, np.array([1.0])))


def test_frechet_none_b_returns_nan():
    assert math.isnan(_frechet(np.array([1.0]), None))


def test_frechet_empty_array_returns_nan():
    assert math.isnan(_frechet(np.array([]), np.array([1.0])))


def test_frechet_different_means():
    a = np.zeros(100)
    b = np.ones(100)
    fd = _frechet(a, b)
    # mu_a=0, mu_b=1, std_a=0, std_b=0  => FD = 1
    assert fd == pytest.approx(1.0)


def test_frechet_different_stds():
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 1.0, 1000)
    b = rng.normal(0.0, 2.0, 1000)
    fd = _frechet(a, b)
    assert fd > 0.0


# --- compute_frechet tests ---

def test_compute_frechet_skips_removed_keys():
    diff = _make_diff(w=_td([1.0], None, status="removed"))
    rows = compute_frechet(diff)
    assert rows == []


def test_compute_frechet_includes_added_keys():
    diff = _make_diff(w=_td(None, [1.0, 2.0], status="added"))
    rows = compute_frechet(diff)
    assert len(rows) == 1
    assert math.isnan(rows[0].frechet)


def test_compute_frechet_sorted_descending_by_fd():
    diff = _make_diff(
        small=_td([0.0, 0.0], [0.1, 0.1]),
        large=_td([0.0, 0.0], [10.0, 10.0]),
    )
    rows = compute_frechet(diff)
    assert rows[0].key == "large"
    assert rows[1].key == "small"


def test_compute_frechet_top_n_limits_rows():
    diff = _make_diff(
        a=_td([0.0], [1.0]),
        b=_td([0.0], [2.0]),
        c=_td([0.0], [3.0]),
    )
    rows = compute_frechet(diff, top_n=2)
    assert len(rows) == 2


# --- format_frechet tests ---

def test_format_frechet_empty_returns_message():
    assert "No" in format_frechet([])


def test_format_frechet_contains_key():
    diff = _make_diff(layer=_td([0.0, 1.0], [1.0, 2.0]))
    rows = compute_frechet(diff)
    out = format_frechet(rows)
    assert "layer" in out


def test_format_frechet_contains_header():
    diff = _make_diff(x=_td([0.0], [1.0]))
    rows = compute_frechet(diff)
    out = format_frechet(rows)
    assert "FD" in out
