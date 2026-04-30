"""Tests for checkpoint_diff.spectral and checkpoint_diff.cli_spectral."""
from __future__ import annotations

import argparse
import math

import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.spectral import (
    SpectralRow,
    _spectral_energy,
    compute_spectral,
    format_spectral,
)
from checkpoint_diff.cli_spectral import add_spectral_args, apply_spectral


def _td(a, b, status="changed"):
    return TensorDiff(
        a=np.array(a, dtype=float) if a is not None else None,
        b=np.array(b, dtype=float) if b is not None else None,
        status=status,
        shape_changed=(np.array(a).shape != np.array(b).shape) if (a is not None and b is not None) else False,
        mean_a=float(np.mean(a)) if a is not None else math.nan,
        mean_b=float(np.mean(b)) if b is not None else math.nan,
        std_a=float(np.std(a)) if a is not None else math.nan,
        std_b=float(np.std(b)) if b is not None else math.nan,
    )


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(tensors=kwargs)


def _make_parser():
    p = argparse.ArgumentParser()
    add_spectral_args(p)
    return p


# --- _spectral_energy ---

def test_spectral_energy_none_returns_nan():
    assert math.isnan(_spectral_energy(None, k=3))


def test_spectral_energy_empty_returns_nan():
    assert math.isnan(_spectral_energy(np.array([]).reshape(0, 4), k=3))


def test_spectral_energy_constant_matrix_is_one_for_k1():
    arr = np.ones((4, 4))
    energy = _spectral_energy(arr, k=1)
    assert abs(energy - 1.0) < 1e-6


def test_spectral_energy_1d_array_treated_as_row():
    arr = np.array([1.0, 2.0, 3.0])
    energy = _spectral_energy(arr, k=1)
    assert 0.0 < energy <= 1.0


def test_spectral_energy_full_rank_k_equals_rank():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((6, 6))
    energy_all = _spectral_energy(arr, k=6)
    assert abs(energy_all - 1.0) < 1e-6


# --- compute_spectral ---

def test_compute_spectral_returns_row_for_changed_key():
    diff = _make_diff(w=_td([[1, 0], [0, 2]], [[2, 0], [0, 1]]))
    rows = compute_spectral(diff, top_k=1)
    assert len(rows) == 1
    assert rows[0].key == "w"


def test_compute_spectral_skips_removed_key():
    diff = _make_diff(w=_td([[1, 0], [0, 2]], None, status="removed"))
    rows = compute_spectral(diff, top_k=1)
    assert rows == []


def test_compute_spectral_skips_unchanged_by_default():
    diff = _make_diff(w=_td([1.0, 2.0], [1.0, 2.0], status="unchanged"))
    rows = compute_spectral(diff, top_k=1, include_unchanged=False)
    assert rows == []


def test_compute_spectral_includes_unchanged_when_flag_set():
    diff = _make_diff(w=_td([1.0, 2.0], [1.0, 2.0], status="unchanged"))
    rows = compute_spectral(diff, top_k=1, include_unchanged=True)
    assert len(rows) == 1


def test_compute_spectral_sorted_by_abs_delta_descending():
    rng = np.random.default_rng(42)
    a1 = rng.standard_normal((8, 8))
    b1 = rng.standard_normal((8, 8))
    a2 = np.eye(8)
    b2 = np.eye(8)
    diff = _make_diff(big=_td(a1.tolist(), b1.tolist()), tiny=_td(a2.tolist(), b2.tolist()))
    rows = compute_spectral(diff, top_k=3)
    deltas = [abs(r.delta) for r in rows if not math.isnan(r.delta)]
    assert deltas == sorted(deltas, reverse=True)


# --- format_spectral ---

def test_format_spectral_empty_returns_message():
    assert "No spectral" in format_spectral([])


def test_format_spectral_contains_key_name():
    diff = _make_diff(layer1=_td([[1, 2], [3, 4]], [[4, 3], [2, 1]]))
    rows = compute_spectral(diff, top_k=1)
    output = format_spectral(rows)
    assert "layer1" in output


def test_format_spectral_contains_header():
    diff = _make_diff(layer1=_td([[1, 2], [3, 4]], [[4, 3], [2, 1]]))
    rows = compute_spectral(diff, top_k=2)
    output = format_spectral(rows)
    assert "energy_a" in output
    assert "energy_b" in output
    assert "delta" in output


# --- CLI ---

def test_add_spectral_args_registers_flag():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.spectral is False


def test_add_spectral_args_top_k_default():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.spectral_top_k == 5


def test_add_spectral_args_top_n_default_none():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.spectral_top_n is None


def test_apply_spectral_returns_none_when_flag_not_set():
    diff = _make_diff(w=_td([1.0, 2.0], [2.0, 3.0]))
    p = _make_parser()
    ns = p.parse_args([])
    assert apply_spectral(ns, diff) is None


def test_apply_spectral_returns_string_when_flag_set():
    diff = _make_diff(w=_td([[1, 2], [3, 4]], [[4, 3], [2, 1]]))
    p = _make_parser()
    ns = p.parse_args(["--spectral"])
    result = apply_spectral(ns, diff)
    assert isinstance(result, str)
    assert "w" in result


def test_apply_spectral_top_n_limits_rows():
    diff = _make_diff(
        a=_td([[1, 0], [0, 2]], [[2, 0], [0, 3]]),
        b=_td([[1, 0], [0, 1]], [[3, 0], [0, 1]]),
        c=_td([[1, 2], [3, 4]], [[4, 3], [2, 1]]),
    )
    p = _make_parser()
    ns = p.parse_args(["--spectral", "--spectral-top-n", "1"])
    result = apply_spectral(ns, diff)
    assert result is not None
    key_lines = [ln for ln in result.splitlines() if not ln.startswith(("key", "-"))]
    assert len(key_lines) == 1
