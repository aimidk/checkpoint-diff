"""Tests for checkpoint_diff.variance and checkpoint_diff.cli_variance."""
from __future__ import annotations

import argparse
import math

import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff, CheckpointDiff
from checkpoint_diff.variance import (
    _variance,
    _cv,
    compute_variance,
    format_variance,
)
from checkpoint_diff.cli_variance import add_variance_args, apply_variance


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _td(a, b, status="changed") -> TensorDiff:
    arr_a = np.array(a, dtype=float) if a is not None else None
    arr_b = np.array(b, dtype=float) if b is not None else None
    return TensorDiff(
        status=status,
        array_a=arr_a,
        array_b=arr_b,
        shape_a=None if arr_a is None else arr_a.shape,
        shape_b=None if arr_b is None else arr_b.shape,
        mean_a=float(np.mean(arr_a)) if arr_a is not None else math.nan,
        mean_b=float(np.mean(arr_b)) if arr_b is not None else math.nan,
        std_a=float(np.std(arr_a)) if arr_a is not None else math.nan,
        std_b=float(np.std(arr_b)) if arr_b is not None else math.nan,
    )


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(tensors=kwargs)


# ---------------------------------------------------------------------------
# _variance
# ---------------------------------------------------------------------------

def test_variance_basic():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert math.isclose(_variance(arr), float(np.var(arr)))


def test_variance_none_returns_nan():
    assert math.isnan(_variance(None))


def test_variance_empty_returns_nan():
    assert math.isnan(_variance(np.array([])))


# ---------------------------------------------------------------------------
# _cv
# ---------------------------------------------------------------------------

def test_cv_basic():
    arr = np.array([2.0, 4.0, 6.0])
    expected = float(np.std(arr) / abs(np.mean(arr)))
    assert math.isclose(_cv(arr), expected)


def test_cv_zero_mean_returns_nan():
    assert math.isnan(_cv(np.array([-1.0, 1.0])))


def test_cv_none_returns_nan():
    assert math.isnan(_cv(None))


# ---------------------------------------------------------------------------
# compute_variance
# ---------------------------------------------------------------------------

def test_compute_variance_returns_row_for_changed_key():
    diff = _make_diff(w=_td([1, 2, 3], [4, 5, 6]))
    rows = compute_variance(diff)
    assert len(rows) == 1
    assert rows[0].key == "w"


def test_compute_variance_skips_removed_key():
    diff = _make_diff(w=_td([1, 2], None, status="removed"))
    rows = compute_variance(diff)
    assert rows == []


def test_compute_variance_skips_unchanged_by_default():
    diff = _make_diff(w=_td([1, 2], [1, 2], status="unchanged"))
    rows = compute_variance(diff)
    assert rows == []


def test_compute_variance_includes_unchanged_when_flag_set():
    diff = _make_diff(w=_td([1, 2], [1, 2], status="unchanged"))
    rows = compute_variance(diff, include_unchanged=True)
    assert len(rows) == 1


def test_compute_variance_sorted_by_abs_delta_descending():
    diff = _make_diff(
        small=_td([1.0, 1.0], [1.0, 1.1]),
        large=_td([0.0, 0.0], [0.0, 10.0]),
    )
    rows = compute_variance(diff)
    assert rows[0].key == "large"


# ---------------------------------------------------------------------------
# format_variance
# ---------------------------------------------------------------------------

def test_format_variance_contains_key():
    diff = _make_diff(layer=_td([1, 2, 3], [2, 3, 4]))
    rows = compute_variance(diff)
    output = format_variance(rows)
    assert "layer" in output


def test_format_variance_empty_returns_message():
    assert "No variance data" in format_variance([])


def test_format_variance_top_n_limits_rows():
    diff = _make_diff(
        a=_td([0, 1], [0, 2]),
        b=_td([0, 1], [0, 3]),
        c=_td([0, 1], [0, 4]),
    )
    rows = compute_variance(diff)
    output = format_variance(rows, top_n=2)
    # Only 2 data lines (plus header + sep)
    data_lines = [l for l in output.splitlines() if l and not l.startswith("-") and not l.startswith("Key")]
    assert len(data_lines) == 2


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_variance_args(p)
    return p


def test_add_variance_args_registers_flag():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.variance is False


def test_add_variance_args_top_n_default_none():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.variance_top_n is None


def test_add_variance_args_include_unchanged_default_false():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.variance_include_unchanged is False


def test_apply_variance_returns_none_when_flag_not_set():
    diff = _make_diff(w=_td([1, 2], [3, 4]))
    p = _make_parser()
    ns = p.parse_args([])
    assert apply_variance(ns, diff) is None


def test_apply_variance_returns_string_when_flag_set():
    diff = _make_diff(w=_td([1, 2], [3, 4]))
    p = _make_parser()
    ns = p.parse_args(["--variance"])
    result = apply_variance(ns, diff)
    assert isinstance(result, str)
    assert "w" in result
