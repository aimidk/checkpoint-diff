"""Tests for checkpoint_diff.norm_ratio."""
from __future__ import annotations

import math

import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.norm_ratio import (
    NormRatioRow,
    _l2,
    _ratio,
    compute_norm_ratios,
    format_norm_ratios,
)


def _td(a, b, status="changed") -> TensorDiff:
    return TensorDiff(
        data_a=np.array(a, dtype=float) if a is not None else None,
        data_b=np.array(b, dtype=float) if b is not None else None,
        status=status,
    )


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(tensors=kwargs)


# --- unit tests for helpers ---

def test_l2_basic():
    assert math.isclose(_l2(np.array([3.0, 4.0])), 5.0)


def test_l2_none_returns_nan():
    assert math.isnan(_l2(None))


def test_l2_empty_returns_nan():
    assert math.isnan(_l2(np.array([])))


def test_ratio_normal():
    assert math.isclose(_ratio(2.0, 6.0), 3.0)


def test_ratio_zero_norm_a_returns_nan():
    assert math.isnan(_ratio(0.0, 5.0))


def test_ratio_nan_inputs_returns_nan():
    assert math.isnan(_ratio(float("nan"), 1.0))


# --- compute_norm_ratios ---

def test_compute_returns_row_per_changed_key():
    diff = _make_diff(
        w1=_td([1.0, 0.0], [2.0, 0.0]),
        w2=_td([0.0, 1.0], [0.0, 3.0]),
    )
    rows = compute_norm_ratios(diff)
    assert len(rows) == 2
    keys = {r.key for r in rows}
    assert keys == {"w1", "w2"}


def test_compute_skips_unchanged_by_default():
    diff = _make_diff(
        w1=_td([1.0], [1.0], status="unchanged"),
        w2=_td([1.0], [2.0], status="changed"),
    )
    rows = compute_norm_ratios(diff)
    assert all(r.key != "w1" for r in rows)


def test_compute_includes_unchanged_when_flag_set():
    diff = _make_diff(
        w1=_td([1.0], [1.0], status="unchanged"),
    )
    rows = compute_norm_ratios(diff, include_unchanged=True)
    assert len(rows) == 1


def test_compute_ratio_correct():
    diff = _make_diff(w=_td([3.0, 4.0], [6.0, 8.0]))
    rows = compute_norm_ratios(diff)
    assert len(rows) == 1
    assert math.isclose(rows[0].norm_a, 5.0)
    assert math.isclose(rows[0].norm_b, 10.0)
    assert math.isclose(rows[0].ratio, 2.0)


def test_compute_sorted_by_deviation_from_one():
    diff = _make_diff(
        small=_td([1.0], [1.1]),   # ratio ~1.1 -> deviation 0.1
        large=_td([1.0], [5.0]),   # ratio 5.0  -> deviation 4.0
    )
    rows = compute_norm_ratios(diff)
    assert rows[0].key == "large"


def test_added_key_norm_a_is_nan():
    diff = _make_diff(w=_td(None, [1.0, 2.0], status="added"))
    rows = compute_norm_ratios(diff)
    assert math.isnan(rows[0].norm_a)
    assert math.isnan(rows[0].ratio)


# --- format_norm_ratios ---

def test_format_contains_key_name():
    diff = _make_diff(my_weight=_td([1.0, 0.0], [0.0, 1.0]))
    rows = compute_norm_ratios(diff)
    out = format_norm_ratios(rows)
    assert "my_weight" in out


def test_format_empty_returns_message():
    assert "No norm ratio data" in format_norm_ratios([])


def test_format_top_n_limits_output():
    diff = _make_diff(
        a=_td([1.0], [2.0]),
        b=_td([1.0], [3.0]),
        c=_td([1.0], [4.0]),
    )
    rows = compute_norm_ratios(diff)
    out = format_norm_ratios(rows, top_n=2)
    # Only 2 data rows (plus header + separator)
    data_lines = [l for l in out.splitlines() if l and not l.startswith("-") and "Key" not in l]
    assert len(data_lines) == 2
