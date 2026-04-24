"""Tests for checkpoint_diff.cli_regression."""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import pytest

from checkpoint_diff.cli_regression import add_regression_args, apply_regression
from checkpoint_diff.diff import CheckpointDiff, TensorDiff


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_regression_args(p)
    return p


def _td(mean_a: float = 1.0, mean_b: float = 5.0) -> TensorDiff:
    return TensorDiff(
        status="changed",
        shape_a=(4,),
        shape_b=(4,),
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=1.0,
        std_b=1.0,
    )


def _make_diff(**kwargs: TensorDiff) -> CheckpointDiff:
    return CheckpointDiff(tensors=kwargs)


# ---------------------------------------------------------------------------

def test_add_regression_args_registers_ref_flag():
    p = _make_parser()
    args = p.parse_args([])
    assert hasattr(args, "regression_ref")
    assert args.regression_ref is None


def test_add_regression_args_tolerance_default():
    p = _make_parser()
    args = p.parse_args([])
    assert args.regression_tolerance == 0.0


def test_add_regression_args_show_all_default():
    p = _make_parser()
    args = p.parse_args([])
    assert args.regression_show_all is False


def test_apply_regression_returns_none_without_flag():
    p = _make_parser()
    args = p.parse_args([])
    result = apply_regression(args, _make_diff(w=_td()))
    assert result is None


def test_apply_regression_returns_string_with_ref(tmp_path):
    ref_path = tmp_path / "ref.npz"
    np.savez(str(ref_path), w=np.array([0.0, 0.0, 0.0, 0.0]))

    p = _make_parser()
    args = p.parse_args(["--regression-ref", str(ref_path)])
    diff = _make_diff(w=_td(mean_a=1.0, mean_b=5.0))
    result = apply_regression(args, diff)
    assert result is not None
    assert isinstance(result, str)


def test_apply_regression_flags_regression(tmp_path):
    ref_path = tmp_path / "ref.npz"
    np.savez(str(ref_path), w=np.zeros(4))

    p = _make_parser()
    args = p.parse_args(["--regression-ref", str(ref_path)])
    diff = _make_diff(w=_td(mean_a=1.0, mean_b=5.0))
    result = apply_regression(args, diff)
    assert "away" in result


def test_apply_regression_show_all_passed_through(tmp_path):
    ref_path = tmp_path / "ref.npz"
    np.savez(str(ref_path), w=np.zeros(4))

    p = _make_parser()
    args = p.parse_args(["--regression-ref", str(ref_path), "--regression-show-all"])
    diff = _make_diff(w=_td(mean_a=5.0, mean_b=1.0))  # toward, not regressed
    result = apply_regression(args, diff)
    assert "w" in result  # visible because show_all=True
