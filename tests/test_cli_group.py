"""Tests for checkpoint_diff.cli_group."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from checkpoint_diff.cli_group import add_group_args, apply_grouping
from checkpoint_diff.diff import TensorDiff


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_group_args(p)
    return p


def _td(status: str = "changed") -> TensorDiff:
    a = np.array([1.0])
    b = np.array([2.0])
    return TensorDiff(key="k", status=status, tensor_a=a, tensor_b=b)


def _make_diff():
    return {
        "enc.weight": _td("changed"),
        "enc.bias": _td("unchanged"),
        "dec.weight": _td("added"),
    }


def test_add_group_args_registers_prefix_flag():
    p = _make_parser()
    args = p.parse_args([])
    assert hasattr(args, "group_by_prefix")


def test_add_group_args_default_prefix_false():
    p = _make_parser()
    args = p.parse_args([])
    assert args.group_by_prefix is False


def test_add_group_args_default_sep_dot():
    p = _make_parser()
    args = p.parse_args([])
    assert args.group_sep == "."


def test_add_group_args_group_map_default_none():
    p = _make_parser()
    args = p.parse_args([])
    assert args.group_map is None


def test_apply_grouping_prefix_returns_groups(capsys):
    p = _make_parser()
    args = p.parse_args(["--group-by-prefix"])
    diff = _make_diff()
    result = apply_grouping(args, diff)
    assert result is not None
    assert "enc" in result
    assert "dec" in result


def test_apply_grouping_prefix_prints_output(capsys):
    p = _make_parser()
    args = p.parse_args(["--group-by-prefix"])
    apply_grouping(args, _make_diff())
    captured = capsys.readouterr()
    assert "enc" in captured.out


def test_apply_grouping_no_flags_returns_none():
    p = _make_parser()
    args = p.parse_args([])
    result = apply_grouping(args, _make_diff())
    assert result is None


def test_apply_grouping_map_json(capsys):
    p = _make_parser()
    mapping = json.dumps({"enc.weight": "backbone", "enc.bias": "backbone"})
    args = p.parse_args(["--group-map", mapping])
    result = apply_grouping(args, _make_diff())
    assert result is not None
    assert "backbone" in result
