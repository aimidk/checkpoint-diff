"""Tests for checkpoint_diff.isotropy and checkpoint_diff.cli_isotropy."""
from __future__ import annotations

import argparse
import math

import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.isotropy import (
    IsotropyRow,
    _isotropy,
    compute_isotropy,
    format_isotropy,
)
from checkpoint_diff.cli_isotropy import add_isotropy_args, apply_isotropy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _td(a, b, status="changed") -> TensorDiff:
    arr_a = np.array(a, dtype=float) if a is not None else None
    arr_b = np.array(b, dtype=float) if b is not None else None
    mean_a = float(np.mean(arr_a)) if arr_a is not None else math.nan
    mean_b = float(np.mean(arr_b)) if arr_b is not None else math.nan
    std_a = float(np.std(arr_a)) if arr_a is not None else math.nan
    std_b = float(np.std(arr_b)) if arr_b is not None else math.nan
    return TensorDiff(
        status=status,
        array_a=arr_a,
        array_b=arr_b,
        shape_a=arr_a.shape if arr_a is not None else None,
        shape_b=arr_b.shape if arr_b is not None else None,
        mean_a=mean_a, mean_b=mean_b,
        std_a=std_a, std_b=std_b,
    )


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(kwargs)


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_isotropy_args(p)
    return p


# ---------------------------------------------------------------------------
# _isotropy unit tests
# ---------------------------------------------------------------------------

def test_isotropy_none_returns_nan():
    assert math.isnan(_isotropy(None))


def test_isotropy_empty_returns_nan():
    assert math.isnan(_isotropy(np.array([])))


def test_isotropy_1d_single_value_returns_nan():
    # 1-element array → only one singular value → entropy = 0 → exp(0)/1 = 1
    # but reshape gives (1,1) which has shape[0] < 2
    assert math.isnan(_isotropy(np.array([5.0])))


def test_isotropy_identity_matrix_is_one():
    # Identity matrix has equal singular values → maximum entropy → score == 1
    arr = np.eye(4)
    score = _isotropy(arr)
    assert not math.isnan(score)
    assert abs(score - 1.0) < 1e-6


def test_isotropy_rank1_matrix_is_low():
    # Rank-1 matrix has all variance in one direction → low isotropy
    arr = np.outer(np.arange(1, 6, dtype=float), np.ones(5))
    score = _isotropy(arr)
    assert not math.isnan(score)
    assert score < 0.5


def test_isotropy_value_in_zero_one():
    rng = np.random.default_rng(42)
    arr = rng.standard_normal((8, 8))
    score = _isotropy(arr)
    assert 0.0 < score <= 1.0


# ---------------------------------------------------------------------------
# compute_isotropy
# ---------------------------------------------------------------------------

def test_compute_isotropy_skips_removed():
    diff = _make_diff(
        w=_td(None, None, status="removed"),
    )
    rows = compute_isotropy(diff)
    assert rows == []


def test_compute_isotropy_returns_row_per_key():
    diff = _make_diff(
        a=_td(np.eye(4), np.eye(4) * 2),
        b=_td(np.eye(3), np.eye(3)),
    )
    rows = compute_isotropy(diff)
    assert len(rows) == 2
    keys = {r.key for r in rows}
    assert keys == {"a", "b"}


def test_compute_isotropy_top_n_limits_results():
    diff = _make_diff(
        a=_td(np.eye(4), np.eye(4) + 0.1),
        b=_td(np.eye(3), np.eye(3) + 0.5),
        c=_td(np.eye(5), np.eye(5) + 1.0),
    )
    rows = compute_isotropy(diff, top_n=2)
    assert len(rows) == 2


def test_compute_isotropy_sorted_by_abs_delta():
    rng = np.random.default_rng(0)
    diff = _make_diff(
        small=_td(np.eye(4), np.eye(4) + rng.standard_normal((4, 4)) * 0.01),
        large=_td(np.eye(4), rng.standard_normal((4, 4)) * 10),
    )
    rows = compute_isotropy(diff)
    assert rows[0].key == "large" or abs(rows[0].delta) >= abs(rows[1].delta)


# ---------------------------------------------------------------------------
# format_isotropy
# ---------------------------------------------------------------------------

def test_format_isotropy_empty():
    assert format_isotropy([]) == "No isotropy data."


def test_format_isotropy_contains_key():
    row = IsotropyRow(key="layer.weight", isotropy_a=0.8, isotropy_b=0.6, delta=-0.2)
    out = format_isotropy([row])
    assert "layer.weight" in out
    assert "0.8000" in out
    assert "0.6000" in out


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def test_add_isotropy_args_registers_flag():
    p = _make_parser()
    args = p.parse_args([])
    assert args.isotropy is False


def test_add_isotropy_args_top_n_default_none():
    p = _make_parser()
    args = p.parse_args([])
    assert args.isotropy_top_n is None


def test_isotropy_flag_parsed():
    p = _make_parser()
    args = p.parse_args(["--isotropy"])
    assert args.isotropy is True


def test_apply_isotropy_returns_none_when_flag_not_set():
    p = _make_parser()
    args = p.parse_args([])
    diff = _make_diff(w=_td(np.eye(3), np.eye(3)))
    result = apply_isotropy(args, diff, print_fn=lambda _: None)
    assert result is None


def test_apply_isotropy_returns_string_when_flag_set():
    p = _make_parser()
    args = p.parse_args(["--isotropy"])
    diff = _make_diff(w=_td(np.eye(3), np.eye(3) + 0.1))
    result = apply_isotropy(args, diff, print_fn=lambda _: None)
    assert isinstance(result, str)
    assert "w" in result
