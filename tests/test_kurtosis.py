"""Tests for checkpoint_diff.kurtosis and checkpoint_diff.cli_kurtosis."""
from __future__ import annotations

import argparse

import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.kurtosis import (
    KurtosisRow,
    _kurtosis,
    compute_kurtosis,
    format_kurtosis,
)
from checkpoint_diff.cli_kurtosis import add_kurtosis_args, apply_kurtosis


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _td(a, b, status="changed") -> TensorDiff:
    return TensorDiff(
        array_a=np.array(a, dtype=float) if a is not None else None,
        array_b=np.array(b, dtype=float) if b is not None else None,
        status=status,
    )


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(tensors=kwargs)


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_kurtosis_args(p)
    return p


# ---------------------------------------------------------------------------
# _kurtosis unit tests
# ---------------------------------------------------------------------------

def test_kurtosis_none_returns_nan():
    assert np.isnan(_kurtosis(None))


def test_kurtosis_constant_array_returns_nan():
    assert np.isnan(_kurtosis(np.ones(10)))


def test_kurtosis_small_array_returns_nan():
    assert np.isnan(_kurtosis(np.array([1.0, 2.0, 3.0])))


def test_kurtosis_normal_approx_near_zero():
    rng = np.random.default_rng(0)
    data = rng.normal(size=100_000)
    k = _kurtosis(data)
    assert abs(k) < 0.1


def test_kurtosis_uniform_is_negative():
    rng = np.random.default_rng(1)
    data = rng.uniform(size=100_000)
    k = _kurtosis(data)
    assert k < 0


# ---------------------------------------------------------------------------
# compute_kurtosis tests
# ---------------------------------------------------------------------------

def test_compute_kurtosis_skips_removed_keys():
    diff = _make_diff(w=_td([1, 2, 3, 4, 5], None, status="removed"))
    rows = compute_kurtosis(diff)
    assert rows == []


def test_compute_kurtosis_skips_unchanged_by_default():
    diff = _make_diff(w=_td([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0], status="unchanged"))
    rows = compute_kurtosis(diff)
    assert rows == []


def test_compute_kurtosis_includes_unchanged_when_flag_set():
    diff = _make_diff(w=_td([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0], status="unchanged"))
    rows = compute_kurtosis(diff, include_unchanged=True)
    assert len(rows) == 1


def test_compute_kurtosis_returns_row_for_changed_key():
    rng = np.random.default_rng(42)
    a = rng.normal(size=500)
    b = rng.normal(size=500)
    diff = _make_diff(layer=_td(a.tolist(), b.tolist()))
    rows = compute_kurtosis(diff)
    assert len(rows) == 1
    assert rows[0].key == "layer"


def test_compute_kurtosis_top_n_limits_output():
    rng = np.random.default_rng(7)
    diff = _make_diff(
        a=_td(rng.normal(size=200).tolist(), rng.normal(size=200).tolist()),
        b=_td(rng.normal(size=200).tolist(), rng.normal(size=200).tolist()),
        c=_td(rng.normal(size=200).tolist(), rng.normal(size=200).tolist()),
    )
    rows = compute_kurtosis(diff, top_n=2)
    assert len(rows) == 2


def test_compute_kurtosis_sorted_by_abs_delta():
    rng = np.random.default_rng(99)
    diff = _make_diff(
        small=_td(rng.normal(size=200).tolist(), rng.normal(size=200).tolist()),
        large=_td(rng.normal(size=200).tolist(), (rng.exponential(size=200) * 5).tolist()),
    )
    rows = compute_kurtosis(diff)
    assert abs(rows[0].delta) >= abs(rows[-1].delta)


# ---------------------------------------------------------------------------
# format_kurtosis tests
# ---------------------------------------------------------------------------

def test_format_kurtosis_empty_returns_message():
    result = format_kurtosis([])
    assert "No kurtosis" in result


def test_format_kurtosis_contains_key():
    rows = [KurtosisRow(key="my.layer", kurtosis_a=0.1, kurtosis_b=-0.5, delta=-0.6, status="changed")]
    result = format_kurtosis(rows)
    assert "my.layer" in result


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

def test_add_kurtosis_args_registers_flag():
    p = _make_parser()
    args = p.parse_args([])
    assert args.kurtosis is False


def test_add_kurtosis_args_top_n_default_none():
    p = _make_parser()
    args = p.parse_args([])
    assert args.kurtosis_top_n is None


def test_add_kurtosis_args_include_unchanged_default_false():
    p = _make_parser()
    args = p.parse_args([])
    assert args.kurtosis_include_unchanged is False


def test_apply_kurtosis_returns_none_when_flag_not_set():
    rng = np.random.default_rng(0)
    diff = _make_diff(w=_td(rng.normal(size=100).tolist(), rng.normal(size=100).tolist()))
    p = _make_parser()
    args = p.parse_args([])
    assert apply_kurtosis(args, diff) is None


def test_apply_kurtosis_returns_string_when_flag_set():
    rng = np.random.default_rng(0)
    diff = _make_diff(w=_td(rng.normal(size=200).tolist(), rng.normal(size=200).tolist()))
    p = _make_parser()
    args = p.parse_args(["--kurtosis"])
    result = apply_kurtosis(args, diff)
    assert isinstance(result, str)
    assert "w" in result
