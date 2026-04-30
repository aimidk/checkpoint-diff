"""Tests for checkpoint_diff.cli_effective_rank."""
from __future__ import annotations

import argparse

import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.cli_effective_rank import add_effective_rank_args, apply_effective_rank


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_effective_rank_args(p)
    return p


def _td(a, b, status="changed"):
    return TensorDiff(
        array_a=np.array(a, dtype=float) if a is not None else None,
        array_b=np.array(b, dtype=float) if b is not None else None,
        status=status,
    )


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(kwargs)


def test_add_effective_rank_args_registers_flag():
    p = _make_parser()
    actions = {a.dest for a in p._actions}
    assert "effective_rank" in actions


def test_add_effective_rank_args_top_n_default_none():
    p = _make_parser()
    args = p.parse_args([])
    assert args.effective_rank_top_n is None


def test_add_effective_rank_args_default_false():
    p = _make_parser()
    args = p.parse_args([])
    assert args.effective_rank is False


def test_effective_rank_flag_parsed():
    p = _make_parser()
    args = p.parse_args(["--effective-rank"])
    assert args.effective_rank is True


def test_effective_rank_top_n_parsed():
    p = _make_parser()
    args = p.parse_args(["--effective-rank", "--effective-rank-top-n", "5"])
    assert args.effective_rank_top_n == 5


def test_apply_effective_rank_returns_none_when_flag_off():
    p = _make_parser()
    args = p.parse_args([])
    diff = _make_diff(w=_td([[1, 0], [0, 1]], [[2, 0], [0, 2]]))
    assert apply_effective_rank(args, diff) is None


def test_apply_effective_rank_returns_string_when_flag_on():
    p = _make_parser()
    args = p.parse_args(["--effective-rank"])
    diff = _make_diff(w=_td([[1, 0], [0, 1]], [[2, 0], [0, 2]]))
    result = apply_effective_rank(args, diff)
    assert isinstance(result, str)
    assert "w" in result


def test_apply_effective_rank_top_n_respected():
    p = _make_parser()
    args = p.parse_args(["--effective-rank", "--effective-rank-top-n", "1"])
    diff = _make_diff(
        w1=_td([[1, 0], [0, 1]], [[2, 0], [0, 2]]),
        w2=_td([[1, 0], [0, 1]], [[3, 0], [0, 3]]),
    )
    result = apply_effective_rank(args, diff)
    assert result is not None
    # Only 1 data row expected after header + separator
    data_lines = [l for l in result.splitlines() if l.startswith("w")]
    assert len(data_lines) == 1
