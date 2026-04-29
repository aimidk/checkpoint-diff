"""Tests for checkpoint_diff.skewness and checkpoint_diff.cli_skewness."""
from __future__ import annotations

import argparse
import math

import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.skewness import (
    SkewnessRow,
    _skewness,
    compute_skewness,
    format_skewness,
)
from checkpoint_diff.cli_skewness import add_skewness_args, apply_skewness


def _td(a, b, status="changed"):
    return TensorDiff(
        status=status,
        a=np.array(a, dtype=float) if a is not None else None,
        b=np.array(b, dtype=float) if b is not None else None,
        mean_a=None, mean_b=None, std_a=None, std_b=None,
        shape_a=None, shape_b=None,
    )


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(tensors=kwargs)


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_skewness_args(p)
    return p


# --- _skewness unit tests ---

def test_skewness_none_returns_nan():
    assert math.isnan(_skewness(None))


def test_skewness_empty_returns_nan():
    assert math.isnan(_skewness(np.array([])))


def test_skewness_constant_array_returns_nan():
    assert math.isnan(_skewness(np.array([3.0, 3.0, 3.0, 3.0])))


def test_skewness_symmetric_distribution_near_zero():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 10_000)
    assert abs(_skewness(data)) < 0.1


def test_skewness_right_skewed_positive():
    data = np.array([1.0, 1.0, 1.0, 1.0, 10.0])
    assert _skewness(data) > 0


# --- compute_skewness tests ---

def test_compute_skewness_returns_row_for_changed_key():
    diff = _make_diff(w=[_td([1, 2, 3, 4, 10], [1, 1, 1, 1, 1])])
    # Fix: tensors should be a dict of key -> TensorDiff
    diff = CheckpointDiff(tensors={"w": _td([1, 2, 3, 4, 10], [1, 1, 1, 1, 1])})
    rows = compute_skewness(diff)
    assert len(rows) == 1
    assert rows[0].key == "w"


def test_compute_skewness_skips_removed_key():
    diff = CheckpointDiff(tensors={
        "w": _td([1, 2, 3], [1, 2, 3]),
        "gone": _td([1, 2, 3], None, status="removed"),
    })
    rows = compute_skewness(diff)
    keys = [r.key for r in rows]
    assert "gone" not in keys


def test_compute_skewness_sorted_by_abs_delta_descending():
    diff = CheckpointDiff(tensors={
        "small": _td([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        "large": _td([1, 1, 1, 1, 10], [1, 2, 3, 4, 5]),
    })
    rows = compute_skewness(diff)
    assert rows[0].key == "large"


# --- format_skewness tests ---

def test_format_skewness_empty_returns_message():
    assert "No skewness" in format_skewness([])


def test_format_skewness_contains_key():
    rows = [SkewnessRow(key="layer.weight", skew_a=0.5, skew_b=1.2, delta=0.7)]
    out = format_skewness(rows)
    assert "layer.weight" in out


def test_format_skewness_top_n_limits_rows():
    rows = [
        SkewnessRow(key=f"k{i}", skew_a=float(i), skew_b=float(i), delta=0.0)
        for i in range(10)
    ]
    out = format_skewness(rows, top_n=3)
    assert out.count("k") == 3


# --- CLI tests ---

def test_add_skewness_args_registers_flag():
    p = _make_parser()
    args = p.parse_args(["--skewness"])
    assert args.skewness is True


def test_add_skewness_args_top_n_default_none():
    p = _make_parser()
    args = p.parse_args([])
    assert args.skewness_top_n is None


def test_apply_skewness_returns_none_when_flag_not_set():
    p = _make_parser()
    args = p.parse_args([])
    diff = CheckpointDiff(tensors={"w": _td([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])})
    assert apply_skewness(args, diff) is None


def test_apply_skewness_returns_string_when_flag_set():
    p = _make_parser()
    args = p.parse_args(["--skewness"])
    diff = CheckpointDiff(tensors={"w": _td([1, 2, 3, 4, 10], [1, 1, 1, 1, 1])})
    result = apply_skewness(args, diff)
    assert isinstance(result, str)
    assert "w" in result
