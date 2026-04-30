"""Tests for checkpoint_diff.cli_mutual_info."""
from __future__ import annotations

import argparse

import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.cli_mutual_info import add_mutual_info_args, apply_mutual_info


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_mutual_info_args(p)
    return p


def _td(a, b, status="changed"):
    return TensorDiff(
        status=status, array_a=a, array_b=b,
        shape_a=getattr(a, "shape", None), shape_b=getattr(b, "shape", None),
        mean_a=None, mean_b=None, std_a=None, std_b=None,
    )


def _make_diff(tensors):
    return CheckpointDiff(tensors=tensors)


def test_add_mutual_info_args_registers_flag():
    p = _make_parser()
    args = p.parse_args([])
    assert args.mutual_info is False


def test_add_mutual_info_args_bins_default():
    p = _make_parser()
    args = p.parse_args([])
    assert args.mi_bins == 64


def test_add_mutual_info_args_top_n_default_none():
    p = _make_parser()
    args = p.parse_args([])
    assert args.mi_top_n is None


def test_mutual_info_flag_parsed():
    p = _make_parser()
    args = p.parse_args(["--mutual-info"])
    assert args.mutual_info is True


def test_mi_bins_parsed():
    p = _make_parser()
    args = p.parse_args(["--mi-bins", "32"])
    assert args.mi_bins == 32


def test_mi_top_n_parsed():
    p = _make_parser()
    args = p.parse_args(["--mi-top-n", "5"])
    assert args.mi_top_n == 5


def test_apply_mutual_info_returns_none_when_flag_off():
    p = _make_parser()
    args = p.parse_args([])
    diff = _make_diff({})
    assert apply_mutual_info(args, diff) is None


def test_apply_mutual_info_returns_string_when_flag_on():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(100)
    b = a + rng.standard_normal(100) * 0.1
    diff = _make_diff({"w": _td(a, b)})
    p = _make_parser()
    args = p.parse_args(["--mutual-info"])
    result = apply_mutual_info(args, diff)
    assert isinstance(result, str)
    assert "w" in result


def test_apply_mutual_info_top_n_limits_rows():
    rng = np.random.default_rng(1)
    tensors = {f"k{i}": _td(rng.standard_normal(80), rng.standard_normal(80)) for i in range(5)}
    diff = _make_diff(tensors)
    p = _make_parser()
    args = p.parse_args(["--mutual-info", "--mi-top-n", "2"])
    result = apply_mutual_info(args, diff)
    # Each key appears once as a data row; count non-header lines with 'k'
    data_lines = [ln for ln in result.splitlines() if ln.startswith("k")]
    assert len(data_lines) == 2
