"""Tests for checkpoint_diff.cli_magnitude."""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff, CheckpointDiff
from checkpoint_diff.cli_magnitude import add_magnitude_args, apply_magnitude


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_magnitude_args(p)
    return p


def _td(a, b, status="changed") -> TensorDiff:
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    return TensorDiff(
        a=a_arr, b=b_arr, status=status,
        mean_a=float(np.mean(a_arr)), mean_b=float(np.mean(b_arr)),
        std_a=float(np.std(a_arr)), std_b=float(np.std(b_arr)),
        shape_a=a_arr.shape, shape_b=b_arr.shape,
    )


def _make_diff() -> CheckpointDiff:
    return CheckpointDiff({"w": _td([1.0, 2.0], [3.0, 4.0])})


# --- argument registration ---

def test_add_magnitude_args_registers_flag():
    p = _make_parser()
    args = p.parse_args([])
    assert args.magnitude is False


def test_add_magnitude_args_top_n_default_none():
    p = _make_parser()
    args = p.parse_args([])
    assert args.magnitude_top_n is None


def test_add_magnitude_args_export_default_none():
    p = _make_parser()
    args = p.parse_args([])
    assert args.magnitude_export is None


def test_magnitude_flag_parsed():
    p = _make_parser()
    args = p.parse_args(["--magnitude"])
    assert args.magnitude is True


def test_magnitude_top_n_parsed():
    p = _make_parser()
    args = p.parse_args(["--magnitude", "--magnitude-top-n", "5"])
    assert args.magnitude_top_n == 5


# --- apply_magnitude ---

def test_apply_magnitude_returns_none_when_flag_not_set():
    p = _make_parser()
    args = p.parse_args([])
    assert apply_magnitude(args, _make_diff()) is None


def test_apply_magnitude_returns_string_when_flag_set():
    p = _make_parser()
    args = p.parse_args(["--magnitude"])
    result = apply_magnitude(args, _make_diff())
    assert isinstance(result, str)
    assert "w" in result


def test_apply_magnitude_export_writes_csv(tmp_path):
    out_file = str(tmp_path / "mag.csv")
    p = _make_parser()
    args = p.parse_args(["--magnitude", "--magnitude-export", out_file])
    apply_magnitude(args, _make_diff())
    assert os.path.exists(out_file)
    with open(out_file) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["key"] == "w"


def test_apply_magnitude_top_n_limits_output():
    diff = CheckpointDiff({
        "a": _td([1, 2], [3, 4]),
        "b": _td([5, 6], [7, 8]),
        "c": _td([9, 10], [11, 12]),
    })
    p = _make_parser()
    args = p.parse_args(["--magnitude", "--magnitude-top-n", "1"])
    result = apply_magnitude(args, diff)
    data_lines = [l for l in result.splitlines() if l and not l.startswith("-") and "key" not in l]
    assert len(data_lines) == 1
