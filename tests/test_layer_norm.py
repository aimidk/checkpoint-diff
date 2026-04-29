"""Tests for checkpoint_diff.layer_norm."""
from __future__ import annotations

import math

import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff
from checkpoint_diff.layer_norm import (
    LayerNormRow,
    _l1,
    _l2,
    compute_layer_norms,
    format_layer_norms,
)


def _td(a, b, status="changed") -> TensorDiff:
    return TensorDiff(
        array_a=np.array(a, dtype=float) if a is not None else None,
        array_b=np.array(b, dtype=float) if b is not None else None,
        status=status,
    )


def _make_diff(*items):
    return dict(items)


# --- unit helpers ---

def test_l1_basic():
    arr = np.array([1.0, -2.0, 3.0])
    assert _l1(arr) == pytest.approx(6.0)


def test_l2_basic():
    arr = np.array([3.0, 4.0])
    assert _l2(arr) == pytest.approx(5.0)


def test_l1_none_returns_nan():
    assert math.isnan(_l1(None))


def test_l2_none_returns_nan():
    assert math.isnan(_l2(None))


def test_l1_empty_returns_nan():
    assert math.isnan(_l1(np.array([])))


def test_l2_empty_returns_nan():
    assert math.isnan(_l2(np.array([])))


# --- compute_layer_norms ---

def test_compute_layer_norms_returns_row_per_key():
    diff = _make_diff(
        ("w", _td([1.0, 2.0], [2.0, 3.0])),
        ("b", _td([0.5], [0.5], status="unchanged")),
    )
    rows = compute_layer_norms(diff)
    assert len(rows) == 2
    assert {r.key for r in rows} == {"w", "b"}


def test_compute_layer_norms_l1_correct():
    diff = _make_diff(("x", _td([1.0, -1.0], [2.0, -2.0])))
    rows = compute_layer_norms(diff)
    assert rows[0].l1_a == pytest.approx(2.0)
    assert rows[0].l1_b == pytest.approx(4.0)


def test_compute_layer_norms_l2_correct():
    diff = _make_diff(("x", _td([3.0, 4.0], [0.0, 5.0])))
    rows = compute_layer_norms(diff)
    assert rows[0].l2_a == pytest.approx(5.0)
    assert rows[0].l2_b == pytest.approx(5.0)


def test_compute_layer_norms_delta_computed():
    diff = _make_diff(("x", _td([1.0], [3.0])))
    rows = compute_layer_norms(diff)
    assert rows[0].l2_delta == pytest.approx(2.0)


def test_compute_layer_norms_sorted_by_abs_l2_delta_descending():
    diff = _make_diff(
        ("small", _td([1.0], [1.1])),
        ("large", _td([0.0], [100.0])),
    )
    rows = compute_layer_norms(diff)
    assert rows[0].key == "large"


def test_added_key_has_nan_for_a():
    diff = _make_diff(("new", _td(None, [1.0, 2.0], status="added")))
    rows = compute_layer_norms(diff)
    assert math.isnan(rows[0].l1_a)
    assert math.isnan(rows[0].l2_a)
    assert not math.isnan(rows[0].l1_b)


# --- format_layer_norms ---

def test_format_layer_norms_contains_key():
    diff = _make_diff(("weight", _td([1.0], [2.0])))
    rows = compute_layer_norms(diff)
    out = format_layer_norms(rows)
    assert "weight" in out


def test_format_layer_norms_empty_returns_message():
    assert "No layer norm data" in format_layer_norms([])


def test_format_layer_norms_top_n_limits_rows():
    diff = _make_diff(
        ("a", _td([0.0], [1.0])),
        ("b", _td([0.0], [2.0])),
        ("c", _td([0.0], [3.0])),
    )
    rows = compute_layer_norms(diff)
    out = format_layer_norms(rows, top_n=2)
    # header + sep + 2 data rows
    data_lines = [ln for ln in out.splitlines() if ln and not ln.startswith("-") and "Key" not in ln]
    assert len(data_lines) == 2
