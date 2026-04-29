"""Tests for checkpoint_diff.weight_norm and checkpoint_diff.cli_weight_norm."""
from __future__ import annotations

import math
import argparse

import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.weight_norm import (
    _linf,
    _frob,
    _delta,
    compute_weight_norms,
    format_weight_norms,
)
from checkpoint_diff.cli_weight_norm import add_weight_norm_args, apply_weight_norm


def _td(a, b, status="changed"):
    return TensorDiff(
        array_a=np.array(a, dtype=float) if a is not None else None,
        array_b=np.array(b, dtype=float) if b is not None else None,
        status=status,
    )


def _make_diff(**kwargs):
    return CheckpointDiff(tensors=kwargs)


# --- unit helpers ---

def test_linf_basic():
    assert _linf(np.array([-3.0, 1.0, 2.0])) == pytest.approx(3.0)


def test_linf_none_returns_nan():
    assert math.isnan(_linf(None))


def test_linf_empty_returns_nan():
    assert math.isnan(_linf(np.array([])))


def test_frob_basic():
    arr = np.array([3.0, 4.0])
    assert _frob(arr) == pytest.approx(5.0)


def test_frob_none_returns_nan():
    assert math.isnan(_frob(None))


def test_delta_simple():
    assert _delta(1.0, 3.0) == pytest.approx(2.0)


def test_delta_nan_propagates():
    assert math.isnan(_delta(math.nan, 1.0))


# --- compute_weight_norms ---

def test_compute_weight_norms_returns_row_per_key():
    diff = _make_diff(
        w1=_td([1.0, 2.0], [1.5, 2.5]),
        w2=_td([0.0], [1.0]),
    )
    rows = compute_weight_norms(diff)
    assert len(rows) == 2


def test_compute_weight_norms_skips_removed():
    diff = _make_diff(
        w1=_td([1.0], None, status="removed"),
        w2=_td([1.0], [2.0]),
    )
    rows = compute_weight_norms(diff)
    assert all(r.key != "w1" for r in rows)


def test_compute_weight_norms_sorted_by_frob_delta():
    diff = _make_diff(
        small=_td([1.0], [1.1]),
        large=_td([1.0, 0.0], [10.0, 0.0]),
    )
    rows = compute_weight_norms(diff)
    assert rows[0].key == "large"


def test_compute_weight_norms_added_key_has_nan_linf_a():
    diff = _make_diff(new=_td(None, [1.0, 2.0], status="added"))
    rows = compute_weight_norms(diff)
    assert len(rows) == 1
    assert math.isnan(rows[0].linf_a)


# --- format_weight_norms ---

def test_format_weight_norms_contains_key():
    diff = _make_diff(layer=_td([1.0, 2.0], [2.0, 3.0]))
    rows = compute_weight_norms(diff)
    text = format_weight_norms(rows)
    assert "layer" in text


def test_format_weight_norms_empty_returns_message():
    assert "No weight norm" in format_weight_norms([])


def test_format_weight_norms_top_n_limits_rows():
    diff = _make_diff(
        a=_td([1.0], [2.0]),
        b=_td([1.0], [3.0]),
        c=_td([1.0], [4.0]),
    )
    rows = compute_weight_norms(diff)
    text = format_weight_norms(rows, top_n=2)
    data_lines = [l for l in text.splitlines() if l and not l.startswith("-") and not l.startswith("Key")]
    assert len(data_lines) == 2


# --- CLI helpers ---

def _make_parser():
    p = argparse.ArgumentParser()
    add_weight_norm_args(p)
    return p


def test_add_weight_norm_args_registers_flag():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.weight_norm is False


def test_add_weight_norm_args_top_n_default_none():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.weight_norm_top_n is None


def test_apply_weight_norm_prints_when_flag_set():
    diff = _make_diff(w=_td([1.0], [2.0]))
    p = _make_parser()
    ns = p.parse_args(["--weight-norm"])
    captured = []
    apply_weight_norm(ns, diff, print_fn=captured.append)
    assert len(captured) == 1
    assert "w" in captured[0]


def test_apply_weight_norm_skips_when_flag_not_set():
    diff = _make_diff(w=_td([1.0], [2.0]))
    p = _make_parser()
    ns = p.parse_args([])
    result = apply_weight_norm(ns, diff)
    assert result is None
