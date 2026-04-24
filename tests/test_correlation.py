"""Tests for checkpoint_diff.correlation."""
from __future__ import annotations

import math
import argparse

import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff, CheckpointDiff
from checkpoint_diff.correlation import (
    CorrelationRow,
    _pearson,
    compute_correlations,
    format_correlations,
)
from checkpoint_diff.cli_correlation import add_correlation_args, apply_correlation


def _td(a, b, status="changed") -> TensorDiff:
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    return TensorDiff(
        status=status,
        array_a=a_arr,
        array_b=b_arr,
        shape_a=a_arr.shape,
        shape_b=b_arr.shape,
        mean_a=float(a_arr.mean()),
        mean_b=float(b_arr.mean()),
        std_a=float(a_arr.std()),
        std_b=float(b_arr.std()),
    )


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(kwargs)


# --- _pearson ---

def test_pearson_identical_arrays_returns_one():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert _pearson(a, a) == pytest.approx(1.0)


def test_pearson_opposite_arrays_returns_minus_one():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 2.0, 1.0])
    assert _pearson(a, b) == pytest.approx(-1.0)


def test_pearson_constant_array_returns_nan():
    a = np.array([5.0, 5.0, 5.0])
    b = np.array([1.0, 2.0, 3.0])
    assert math.isnan(_pearson(a, b))


def test_pearson_size_mismatch_returns_nan():
    assert math.isnan(_pearson(np.array([1.0, 2.0]), np.array([1.0])))


# --- compute_correlations ---

def test_compute_correlations_skips_removed_key():
    td = TensorDiff(
        status="removed", array_a=np.array([1.0, 2.0]), array_b=None,
        shape_a=(2,), shape_b=None,
        mean_a=1.5, mean_b=None, std_a=0.5, std_b=None,
    )
    diff = _make_diff(w=td)
    rows = compute_correlations(diff)
    assert rows == []


def test_compute_correlations_includes_changed_key():
    diff = _make_diff(w=_td([1, 2, 3], [3, 2, 1]))
    rows = compute_correlations(diff)
    assert len(rows) == 1
    assert rows[0].key == "w"
    assert rows[0].pearson == pytest.approx(-1.0)


def test_compute_correlations_excludes_unchanged_by_default():
    diff = _make_diff(w=_td([1, 2, 3], [1, 2, 3], status="unchanged"))
    rows = compute_correlations(diff)
    assert rows == []


def test_compute_correlations_includes_unchanged_when_flag_set():
    diff = _make_diff(w=_td([1, 2, 3], [1, 2, 3], status="unchanged"))
    rows = compute_correlations(diff, include_unchanged=True)
    assert len(rows) == 1


def test_compute_correlations_sorted_ascending_by_pearson():
    diff = _make_diff(
        high=_td([1, 2, 3], [1, 2, 3]),
        low=_td([1, 2, 3], [3, 2, 1]),
    )
    rows = compute_correlations(diff)
    assert rows[0].pearson <= rows[1].pearson


# --- format_correlations ---

def test_format_correlations_empty_returns_message():
    assert "No correlation" in format_correlations([])


def test_format_correlations_contains_key():
    rows = [CorrelationRow(key="layer.weight", pearson=0.95, status="changed")]
    out = format_correlations(rows)
    assert "layer.weight" in out
    assert "0.9500" in out


def test_format_correlations_top_n_limits_rows():
    rows = [CorrelationRow(key=f"k{i}", pearson=float(i) / 10, status="changed") for i in range(5)]
    out = format_correlations(rows, top_n=2)
    assert out.count("k") == 2


# --- CLI integration ---

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_correlation_args(p)
    return p


def test_add_correlation_args_registers_flag():
    p = _make_parser()
    args = p.parse_args(["--correlation"])
    assert args.correlation is True


def test_add_correlation_args_defaults():
    p = _make_parser()
    args = p.parse_args([])
    assert args.correlation is False
    assert args.correlation_include_unchanged is False
    assert args.correlation_top_n is None


def test_apply_correlation_no_flag_returns_empty():
    p = _make_parser()
    args = p.parse_args([])
    diff = _make_diff(w=_td([1, 2, 3], [3, 2, 1]))
    result = apply_correlation(args, diff)
    assert result == []


def test_apply_correlation_with_flag_returns_rows(capsys):
    p = _make_parser()
    args = p.parse_args(["--correlation"])
    diff = _make_diff(w=_td([1, 2, 3], [3, 2, 1]))
    rows = apply_correlation(args, diff)
    assert len(rows) == 1
    captured = capsys.readouterr()
    assert "w" in captured.out
