"""Tests for checkpoint_diff.quantile_shift."""
from __future__ import annotations

import math
import argparse

import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.quantile_shift import (
    _quantile_values,
    compute_quantile_shifts,
    format_quantile_shifts,
)
from checkpoint_diff.cli_quantile_shift import add_quantile_shift_args, apply_quantile_shift


def _td(status, a=None, b=None):
    return TensorDiff(
        status=status,
        array_a=np.array(a, dtype=float) if a is not None else None,
        array_b=np.array(b, dtype=float) if b is not None else None,
        shape_a=None if a is None else (len(a),),
        shape_b=None if b is None else (len(b),),
        mean_a=None,
        mean_b=None,
        std_a=None,
        std_b=None,
    )


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(tensors=kwargs)


# --- _quantile_values ---

def test_quantile_values_none_returns_nans():
    result = _quantile_values(None)
    assert all(math.isnan(v) for v in result)


def test_quantile_values_basic():
    arr = np.arange(101, dtype=float)
    result = _quantile_values(arr)
    assert len(result) == 5
    assert abs(result[2] - 50.0) < 1.0  # median near 50


def test_quantile_values_empty_returns_nans():
    result = _quantile_values(np.array([]))
    assert all(math.isnan(v) for v in result)


# --- compute_quantile_shifts ---

def test_compute_skips_removed_keys():
    diff = _make_diff(w=_td("removed", a=[1, 2, 3]))
    rows = compute_quantile_shifts(diff)
    assert rows == []


def test_compute_skips_unchanged_by_default():
    diff = _make_diff(w=_td("unchanged", a=[1, 2, 3], b=[1, 2, 3]))
    rows = compute_quantile_shifts(diff)
    assert rows == []


def test_compute_includes_unchanged_when_flag_set():
    diff = _make_diff(w=_td("unchanged", a=[1, 2, 3], b=[1, 2, 3]))
    rows = compute_quantile_shifts(diff, include_unchanged=True)
    assert len(rows) == 1
    assert rows[0].key == "w"


def test_compute_changed_key_included():
    diff = _make_diff(w=_td("changed", a=[0, 1, 2, 3, 4], b=[10, 11, 12, 13, 14]))
    rows = compute_quantile_shifts(diff)
    assert len(rows) == 1
    assert rows[0].max_abs_shift > 0


def test_compute_sorted_by_max_shift_descending():
    diff = _make_diff(
        small=_td("changed", a=[0, 1, 2, 3, 4], b=[0.1, 1.1, 2.1, 3.1, 4.1]),
        large=_td("changed", a=[0, 1, 2, 3, 4], b=[100, 101, 102, 103, 104]),
    )
    rows = compute_quantile_shifts(diff)
    assert rows[0].key == "large"


def test_compute_top_n_limits_rows():
    diff = _make_diff(
        a=_td("changed", a=[0, 1], b=[10, 11]),
        b=_td("changed", a=[0, 1], b=[5, 6]),
        c=_td("changed", a=[0, 1], b=[1, 2]),
    )
    rows = compute_quantile_shifts(diff, top_n=2)
    assert len(rows) == 2


# --- format_quantile_shifts ---

def test_format_empty_returns_message():
    assert format_quantile_shifts([]) == "No quantile shift data."


def test_format_contains_key_name():
    diff = _make_diff(my_weight=_td("changed", a=[0, 1, 2, 3, 4], b=[5, 6, 7, 8, 9]))
    rows = compute_quantile_shifts(diff)
    output = format_quantile_shifts(rows)
    assert "my_weight" in output


# --- CLI ---

def _make_parser():
    p = argparse.ArgumentParser()
    add_quantile_shift_args(p)
    return p


def test_add_quantile_shift_args_registers_flag():
    p = _make_parser()
    ns = p.parse_args(["--quantile-shift"])
    assert ns.quantile_shift is True


def test_add_quantile_shift_args_top_n_default_none():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.quantile_shift_top_n is None


def test_apply_quantile_shift_returns_none_when_flag_unset():
    diff = _make_diff(w=_td("changed", a=[0, 1, 2], b=[3, 4, 5]))
    p = _make_parser()
    ns = p.parse_args([])
    assert apply_quantile_shift(ns, diff) is None


def test_apply_quantile_shift_returns_string_when_flag_set():
    diff = _make_diff(w=_td("changed", a=[0, 1, 2, 3, 4], b=[5, 6, 7, 8, 9]))
    p = _make_parser()
    ns = p.parse_args(["--quantile-shift"])
    result = apply_quantile_shift(ns, diff)
    assert isinstance(result, str)
    assert "w" in result
