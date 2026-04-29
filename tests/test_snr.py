"""Tests for checkpoint_diff.snr."""
from __future__ import annotations

import math

import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff
from checkpoint_diff.snr import SNRRow, _snr_db, compute_snr, format_snr


def _td(a, b, status="changed"):
    return TensorDiff(
        array_a=np.array(a, dtype=float) if a is not None else None,
        array_b=np.array(b, dtype=float) if b is not None else None,
        status=status,
    )


def _make_diff(*pairs):
    """pairs: (key, TensorDiff)"""
    return dict(pairs)


# --- _snr_db unit tests ---

def test_snr_db_none_returns_nan():
    assert math.isnan(_snr_db(None))


def test_snr_db_empty_returns_nan():
    assert math.isnan(_snr_db(np.array([])))


def test_snr_db_constant_nonzero_returns_inf():
    result = _snr_db(np.array([5.0, 5.0, 5.0]))
    assert math.isinf(result) and result > 0


def test_snr_db_all_zeros_returns_nan():
    assert math.isnan(_snr_db(np.array([0.0, 0.0])))


def test_snr_db_typical_array_finite():
    rng = np.random.default_rng(0)
    arr = rng.normal(loc=5.0, scale=1.0, size=100)
    result = _snr_db(arr)
    assert math.isfinite(result)
    assert result > 0  # signal >> noise in this case


# --- compute_snr ---

def test_compute_snr_skips_unchanged():
    diff = _make_diff(
        ("w", _td([1.0, 2.0], [1.0, 2.0], status="unchanged")),
    )
    rows = compute_snr(diff)
    assert rows == []


def test_compute_snr_returns_row_for_changed_key():
    diff = _make_diff(
        ("layer.weight", _td([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0])),
    )
    rows = compute_snr(diff)
    assert len(rows) == 1
    assert rows[0].key == "layer.weight"
    assert rows[0].status == "changed"


def test_compute_snr_added_key_snr_a_is_nan():
    diff = _make_diff(
        ("new_layer", _td(None, [1.0, 2.0, 3.0], status="added")),
    )
    rows = compute_snr(diff)
    assert len(rows) == 1
    assert math.isnan(rows[0].snr_a)
    assert not math.isnan(rows[0].snr_b)
    assert math.isnan(rows[0].snr_delta)


def test_compute_snr_sorted_by_abs_delta_descending():
    diff = _make_diff(
        ("small", _td([1.0] * 10, [1.01] * 10)),
        ("large", _td([1.0] * 10, [100.0] * 10)),
    )
    rows = compute_snr(diff)
    assert rows[0].key == "large"


# --- format_snr ---

def test_format_snr_empty_returns_message():
    assert "No SNR" in format_snr([])


def test_format_snr_contains_key():
    rows = [SNRRow(key="fc.weight", snr_a=10.0, snr_b=8.5, snr_delta=-1.5, status="changed")]
    out = format_snr(rows)
    assert "fc.weight" in out
    assert "10.00" in out
    assert "-1.50" in out


def test_format_snr_top_n_limits_rows():
    rows = [SNRRow(key=f"k{i}", snr_a=1.0, snr_b=1.0, snr_delta=0.0, status="changed") for i in range(10)]
    out = format_snr(rows, top_n=3)
    assert out.count("changed") == 3
