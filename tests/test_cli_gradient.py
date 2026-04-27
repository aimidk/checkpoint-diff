"""Tests for checkpoint_diff.cli_gradient."""
from __future__ import annotations

import argparse
import math

import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff
from checkpoint_diff.cli_gradient import add_gradient_args, apply_gradient


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_gradient_args(p)
    return p


def _td(a, b) -> TensorDiff:
    arr_a = np.array(a, dtype=float) if a is not None else None
    arr_b = np.array(b, dtype=float) if b is not None else None
    status = "changed" if (a is not None and b is not None) else ("added" if a is None else "removed")
    return TensorDiff(
        status=status,
        array_a=arr_a,
        array_b=arr_b,
        shape_a=None if arr_a is None else arr_a.shape,
        shape_b=None if arr_b is None else arr_b.shape,
        mean_a=None if arr_a is None else float(arr_a.mean()),
        mean_b=None if arr_b is None else float(arr_b.mean()),
        std_a=None if arr_a is None else float(arr_a.std()),
        std_b=None if arr_b is None else float(arr_b.std()),
    )


def _make_diff():
    return {"layer.weight": _td([1.0, 2.0], [2.0, 4.0])}


def test_add_gradient_args_registers_flag():
    p = _make_parser()
    args = p.parse_args([])
    assert args.gradient is False


def test_add_gradient_args_top_n_default_none():
    p = _make_parser()
    args = p.parse_args([])
    assert args.gradient_top_n is None


def test_add_gradient_args_threshold_default_none():
    p = _make_parser()
    args = p.parse_args([])
    assert args.gradient_threshold is None


def test_apply_gradient_returns_none_when_flag_not_set():
    p = _make_parser()
    args = p.parse_args([])
    result = apply_gradient(args, _make_diff())
    assert result is None


def test_apply_gradient_returns_string_when_flag_set():
    p = _make_parser()
    args = p.parse_args(["--gradient"])
    result = apply_gradient(args, _make_diff())
    assert isinstance(result, str)
    assert "layer.weight" in result


def test_apply_gradient_top_n_limits_output():
    diff = {
        "a": _td([1.0], [2.0]),
        "b": _td([1.0], [3.0]),
        "c": _td([1.0], [4.0]),
    }
    p = _make_parser()
    args = p.parse_args(["--gradient", "--gradient-top-n", "1"])
    result = apply_gradient(args, diff)
    assert result is not None
    # Only one data row beyond header
    data_lines = [l for l in result.splitlines() if l and not l.startswith("-") and "Key" not in l]
    assert len(data_lines) == 1


def test_apply_gradient_threshold_filters_small_deltas():
    diff = {
        "tiny": _td([1.0], [1.0001]),
        "big": _td([0.0, 0.0], [100.0, 100.0]),
    }
    p = _make_parser()
    args = p.parse_args(["--gradient", "--gradient-threshold", "50"])
    result = apply_gradient(args, diff)
    assert result is not None
    assert "big" in result
    assert "tiny" not in result
