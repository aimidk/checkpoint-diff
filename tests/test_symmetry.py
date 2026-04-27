"""Tests for checkpoint_diff.symmetry."""
from __future__ import annotations

import math
import pytest

from checkpoint_diff.diff import TensorDiff, CheckpointDiff
from checkpoint_diff.symmetry import (
    SymmetryRow,
    _mean_symmetry,
    _std_ratio,
    compute_symmetry,
    format_symmetry,
)


def _td(
    status: str = "changed",
    mean_a: float = 0.0,
    mean_b: float = 0.0,
    std_a: float = 1.0,
    std_b: float = 1.0,
) -> TensorDiff:
    return TensorDiff(
        status=status,
        shape_a=(4,),
        shape_b=(4,),
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
    )


def _make_diff(**kwargs: TensorDiff) -> CheckpointDiff:
    return CheckpointDiff(kwargs)


# --- unit helpers ---

def test_mean_symmetry_perfect_negatives():
    # mean_a = 1.0, mean_b = -1.0 => sum = 0 => score = 1.0
    assert _mean_symmetry(1.0, -1.0) == pytest.approx(1.0)


def test_mean_symmetry_identical_sign():
    # mean_a = mean_b = 1.0 => sum = 2.0, denom = 2.0 => score = 0.0
    assert _mean_symmetry(1.0, 1.0) == pytest.approx(0.0, abs=1e-6)


def test_std_ratio_equal_stds():
    assert _std_ratio(2.0, 2.0) == pytest.approx(1.0)


def test_std_ratio_unequal_stds():
    ratio = _std_ratio(1.0, 2.0)
    assert 0.0 < ratio < 1.0


# --- compute_symmetry ---

def test_compute_symmetry_skips_removed():
    diff = _make_diff(w=_td(status="removed", mean_a=1.0, mean_b=-1.0))
    rows = compute_symmetry(diff)
    assert rows == []


def test_compute_symmetry_skips_none_means():
    td = TensorDiff(status="changed", shape_a=(2,), shape_b=(2,),
                    mean_a=None, mean_b=None, std_a=None, std_b=None)
    diff = _make_diff(w=td)
    rows = compute_symmetry(diff)
    assert rows == []


def test_compute_symmetry_detects_symmetric_pair():
    diff = _make_diff(layer=_td(mean_a=1.0, mean_b=-1.0, std_a=0.5, std_b=0.5))
    rows = compute_symmetry(diff, mean_threshold=0.9, std_threshold=0.9)
    assert len(rows) == 1
    assert rows[0].is_symmetric is True


def test_compute_symmetry_non_symmetric_pair():
    diff = _make_diff(layer=_td(mean_a=1.0, mean_b=1.0, std_a=0.5, std_b=0.5))
    rows = compute_symmetry(diff, mean_threshold=0.9, std_threshold=0.9)
    assert len(rows) == 1
    assert rows[0].is_symmetric is False


def test_compute_symmetry_sorted_descending():
    diff = _make_diff(
        low=_td(mean_a=1.0, mean_b=0.5),
        high=_td(mean_a=1.0, mean_b=-0.99),
    )
    rows = compute_symmetry(diff)
    assert rows[0].mean_symmetry >= rows[1].mean_symmetry


# --- format_symmetry ---

def test_format_symmetry_empty():
    assert "No symmetry" in format_symmetry([])


def test_format_symmetry_contains_key():
    diff = _make_diff(mykey=_td(mean_a=1.0, mean_b=-1.0, std_a=1.0, std_b=1.0))
    rows = compute_symmetry(diff)
    output = format_symmetry(rows)
    assert "mykey" in output
    assert "MeanSym" in output
