"""Tests for checkpoint_diff.momentum and checkpoint_diff.cli_momentum."""
from __future__ import annotations

import argparse
import math

import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff, CheckpointDiff
from checkpoint_diff.momentum import (
    MomentumRow,
    _l2,
    _rel,
    compute_momentum,
    format_momentum,
)
from checkpoint_diff.cli_momentum import add_momentum_args, apply_momentum


def _td(a, b, status="changed") -> TensorDiff:
    return TensorDiff(
        array_a=np.array(a, dtype=float) if a is not None else None,
        array_b=np.array(b, dtype=float) if b is not None else None,
        status=status,
    )


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(kwargs)


# --- unit helpers ---

def test_l2_basic():
    arr = np.array([3.0, 4.0])
    assert math.isclose(_l2(arr), 5.0)


def test_l2_none_returns_nan():
    assert math.isnan(_l2(None))


def test_l2_empty_returns_nan():
    assert math.isnan(_l2(np.array([])))


def test_rel_zero_base_returns_nan():
    assert math.isnan(_rel(1.0, 0.0))


def test_rel_nan_propagates():
    assert math.isnan(_rel(math.nan, 1.0))
    assert math.isnan(_rel(1.0, math.nan))


def test_rel_basic():
    assert math.isclose(_rel(2.0, 4.0), 0.5)


# --- compute_momentum ---

def test_compute_momentum_changed_key():
    diff = _make_diff(w=_td([1.0, 0.0], [2.0, 0.0]))
    rows = compute_momentum(diff)
    assert len(rows) == 1
    r = rows[0]
    assert r.key == "w"
    assert math.isclose(r.delta_l2, 1.0)
    assert math.isclose(r.base_l2, 1.0)
    assert math.isclose(r.rel_momentum, 1.0)


def test_compute_momentum_skips_removed_key():
    diff = _make_diff(w=_td([1.0], None, status="removed"))
    rows = compute_momentum(diff)
    assert rows == []


def test_compute_momentum_added_key_has_nan_base():
    diff = _make_diff(w=_td(None, [3.0, 4.0], status="added"))
    rows = compute_momentum(diff)
    assert len(rows) == 1
    assert math.isnan(rows[0].rel_momentum)


def test_compute_momentum_sorted_by_rel_momentum_descending():
    diff = _make_diff(
        small=_td([10.0], [10.1]),
        large=_td([1.0], [3.0]),
    )
    rows = compute_momentum(diff)
    assert rows[0].key == "large"


def test_compute_momentum_shape_mismatch_uses_b_as_delta():
    diff = _make_diff(w=_td([1.0, 2.0], [5.0, 5.0, 5.0]))
    rows = compute_momentum(diff)
    assert len(rows) == 1
    # delta_l2 == l2([5,5,5])
    assert math.isclose(rows[0].delta_l2, _l2(np.array([5.0, 5.0, 5.0])))


# --- format_momentum ---

def test_format_momentum_contains_key_name():
    diff = _make_diff(layer=_td([1.0], [2.0]))
    rows = compute_momentum(diff)
    report = format_momentum(rows)
    assert "layer" in report


def test_format_momentum_empty_returns_message():
    assert "No momentum" in format_momentum([])


def test_format_momentum_top_n_limits_rows():
    diff = _make_diff(
        a=_td([1.0], [2.0]),
        b=_td([3.0], [4.0]),
        c=_td([5.0], [6.0]),
    )
    rows = compute_momentum(diff)
    report = format_momentum(rows, top_n=1)
    # Only one data row beyond header/sep
    data_lines = [l for l in report.splitlines() if l and not l.startswith("-") and not l.startswith("Key")]
    assert len(data_lines) == 1


# --- CLI ---

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_momentum_args(p)
    return p


def test_add_momentum_args_registers_flag():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.momentum is False


def test_add_momentum_args_top_n_default_none():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.momentum_top_n is None


def test_add_momentum_args_export_default_none():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.momentum_export is None


def test_apply_momentum_returns_none_when_flag_not_set():
    diff = _make_diff(w=_td([1.0], [2.0]))
    p = _make_parser()
    ns = p.parse_args([])
    assert apply_momentum(ns, diff) is None


def test_apply_momentum_returns_string_when_flag_set():
    diff = _make_diff(w=_td([1.0], [2.0]))
    p = _make_parser()
    ns = p.parse_args(["--momentum"])
    result = apply_momentum(ns, diff)
    assert isinstance(result, str)
    assert "w" in result


def test_apply_momentum_writes_csv(tmp_path):
    diff = _make_diff(w=_td([1.0], [2.0]))
    out = tmp_path / "mom.csv"
    p = _make_parser()
    ns = p.parse_args(["--momentum", "--momentum-export", str(out)])
    apply_momentum(ns, diff)
    assert out.exists()
    content = out.read_text()
    assert "key" in content
    assert "w" in content
