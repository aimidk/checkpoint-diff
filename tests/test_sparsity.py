"""Tests for checkpoint_diff.sparsity and checkpoint_diff.cli_sparsity."""
from __future__ import annotations

import argparse

import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.sparsity import SparsityRow, compute_sparsity, format_sparsity
from checkpoint_diff.cli_sparsity import add_sparsity_args, apply_sparsity


def _td(a, b, status="changed"):
    return TensorDiff(
        tensor_a=np.array(a, dtype=float) if a is not None else None,
        tensor_b=np.array(b, dtype=float) if b is not None else None,
        status=status,
    )


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(kwargs)


# --- compute_sparsity ---

def test_compute_sparsity_returns_row_per_key():
    diff = _make_diff(
        layer=_td([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
    )
    rows = compute_sparsity(diff)
    assert len(rows) == 1
    assert rows[0].key == "layer"


def test_sparsity_values_correct():
    diff = _make_diff(
        w=_td([0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]),
    )
    rows = compute_sparsity(diff)
    assert rows[0].sparsity_a == pytest.approx(0.5)
    assert rows[0].sparsity_b == pytest.approx(0.75)
    assert rows[0].delta_sparsity == pytest.approx(0.25)


def test_near_zero_uses_eps():
    diff = _make_diff(
        w=_td([0.0, 5e-7, 1.0], [0.0, 2e-5, 1.0]),
    )
    rows = compute_sparsity(diff, eps=1e-6)
    # 5e-7 < 1e-6, so near_zero_a = 2/3
    assert rows[0].near_zero_a == pytest.approx(2 / 3)
    # 2e-5 >= 1e-6, so near_zero_b = 1/3
    assert rows[0].near_zero_b == pytest.approx(1 / 3)


def test_unchanged_excluded_by_default():
    diff = _make_diff(
        w=_td([0.0, 1.0], [0.0, 1.0], status="unchanged"),
        b=_td([0.0, 0.0], [1.0, 1.0], status="changed"),
    )
    rows = compute_sparsity(diff)
    keys = [r.key for r in rows]
    assert "w" not in keys
    assert "b" in keys


def test_unchanged_included_when_flag_set():
    diff = _make_diff(
        w=_td([0.0, 1.0], [0.0, 1.0], status="unchanged"),
    )
    rows = compute_sparsity(diff, include_unchanged=True)
    assert any(r.key == "w" for r in rows)


def test_added_key_sparsity_a_is_none():
    diff = _make_diff(
        new_layer=_td(None, [0.0, 1.0, 0.0], status="added"),
    )
    rows = compute_sparsity(diff)
    assert rows[0].sparsity_a is None
    assert rows[0].sparsity_b == pytest.approx(2 / 3)
    assert rows[0].delta_sparsity is None


def test_rows_sorted_by_abs_delta_descending():
    diff = _make_diff(
        small=_td([0.0, 1.0], [0.0, 0.0]),  # delta = 0.5
        large=_td([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]),  # delta = 1.0
    )
    rows = compute_sparsity(diff)
    assert rows[0].key == "large"


# --- format_sparsity ---

def test_format_sparsity_contains_key():
    diff = _make_diff(weight=_td([0.0, 1.0], [0.0, 0.0]))
    rows = compute_sparsity(diff)
    report = format_sparsity(rows)
    assert "weight" in report


def test_format_sparsity_empty_returns_message():
    assert "No sparsity data" in format_sparsity([])


def test_format_sparsity_top_n_limits_rows():
    diff = _make_diff(
        a=_td([0.0], [1.0]),
        b=_td([0.0, 0.0], [0.0, 1.0]),
        c=_td([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    )
    rows = compute_sparsity(diff)
    report = format_sparsity(rows, top_n=1)
    lines = [l for l in report.splitlines() if l and not l.startswith("-") and "Key" not in l]
    assert len(lines) == 1


# --- CLI integration ---

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_sparsity_args(p)
    return p


def test_add_sparsity_args_registers_flag():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.sparsity is False


def test_add_sparsity_args_eps_default():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.sparsity_eps == pytest.approx(1e-6)


def test_add_sparsity_args_top_n_default_none():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.sparsity_top_n is None


def test_apply_sparsity_returns_none_when_flag_off():
    p = _make_parser()
    ns = p.parse_args([])
    diff = _make_diff(w=_td([0.0], [1.0]))
    assert apply_sparsity(ns, diff) is None


def test_apply_sparsity_returns_string_when_flag_on():
    p = _make_parser()
    ns = p.parse_args(["--sparsity"])
    diff = _make_diff(w=_td([0.0, 1.0], [0.0, 0.0]))
    result = apply_sparsity(ns, diff)
    assert isinstance(result, str)
    assert "w" in result
