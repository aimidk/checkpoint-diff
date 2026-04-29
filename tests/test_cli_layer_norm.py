"""Tests for checkpoint_diff.cli_layer_norm."""
from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff
from checkpoint_diff.cli_layer_norm import add_layer_norm_args, apply_layer_norm


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_layer_norm_args(p)
    return p


def _td(a, b, status="changed") -> TensorDiff:
    return TensorDiff(
        array_a=np.array(a, dtype=float) if a is not None else None,
        array_b=np.array(b, dtype=float) if b is not None else None,
        status=status,
    )


def _make_diff(*items):
    return dict(items)


# --- argument registration ---

def test_add_layer_norm_args_registers_flag():
    p = _make_parser()
    args = p.parse_args(["--layer-norm"])
    assert args.layer_norm is True


def test_add_layer_norm_args_default_false():
    p = _make_parser()
    args = p.parse_args([])
    assert args.layer_norm is False


def test_add_layer_norm_args_top_n_default_none():
    p = _make_parser()
    args = p.parse_args([])
    assert args.layer_norm_top_n is None


def test_add_layer_norm_args_top_n_parsed():
    p = _make_parser()
    args = p.parse_args(["--layer-norm", "--layer-norm-top-n", "5"])
    assert args.layer_norm_top_n == 5


def test_add_layer_norm_args_export_default_none():
    p = _make_parser()
    args = p.parse_args([])
    assert args.layer_norm_export is None


# --- apply_layer_norm ---

def test_apply_layer_norm_disabled_returns_none():
    p = _make_parser()
    args = p.parse_args([])
    diff = _make_diff(("w", _td([1.0], [2.0])))
    assert apply_layer_norm(args, diff) is None


def test_apply_layer_norm_enabled_returns_string():
    p = _make_parser()
    args = p.parse_args(["--layer-norm"])
    diff = _make_diff(("w", _td([1.0], [2.0])))
    result = apply_layer_norm(args, diff)
    assert isinstance(result, str)
    assert "w" in result


def test_apply_layer_norm_top_n_respected():
    p = _make_parser()
    args = p.parse_args(["--layer-norm", "--layer-norm-top-n", "1"])
    diff = _make_diff(
        ("a", _td([0.0], [1.0])),
        ("b", _td([0.0], [10.0])),
    )
    result = apply_layer_norm(args, diff)
    # Only one data row should appear
    data_lines = [ln for ln in result.splitlines() if ln and not ln.startswith("-") and "Key" not in ln]
    assert len(data_lines) == 1


def test_apply_layer_norm_export_creates_csv(tmp_path):
    out_file = str(tmp_path / "norms.csv")
    p = _make_parser()
    args = p.parse_args(["--layer-norm", "--layer-norm-export", out_file])
    diff = _make_diff(("w", _td([3.0, 4.0], [0.0, 5.0])))
    apply_layer_norm(args, diff)
    assert os.path.exists(out_file)
    with open(out_file, newline="") as fh:
        reader = list(csv.reader(fh))
    assert reader[0] == ["key", "status", "l1_a", "l1_b", "l2_a", "l2_b", "l1_delta", "l2_delta"]
    assert reader[1][0] == "w"
