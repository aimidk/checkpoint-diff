"""Tests for checkpoint_diff.diff_score and checkpoint_diff.cli_diff_score."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff
from checkpoint_diff.diff_score import DiffScore, compute_score, format_score
from checkpoint_diff.cli_diff_score import add_diff_score_args, apply_diff_score


def _td(status: str, mean_a=None, mean_b=None, std_a=None, std_b=None) -> TensorDiff:
    return TensorDiff(
        status=status,
        shape_a=(4,) if mean_a is not None else None,
        shape_b=(4,) if mean_b is not None else None,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a or 0.0,
        std_b=std_b or 0.0,
    )


def _make_diff(**kwargs) -> dict:
    return kwargs


# ---------------------------------------------------------------------------
# compute_score
# ---------------------------------------------------------------------------

def test_score_all_unchanged_returns_zero_score():
    diff = {"w": _td("unchanged", 1.0, 1.0)}
    ds = compute_score(diff)
    assert ds.score == 0.0
    assert ds.changed_keys == 0


def test_score_counts_changed_added_removed():
    diff = {
        "a": _td("changed", 0.0, 1.0),
        "b": _td("added", None, 2.0),
        "c": _td("removed", 3.0, None),
        "d": _td("unchanged", 1.0, 1.0),
    }
    ds = compute_score(diff)
    assert ds.changed_keys == 1
    assert ds.added_keys == 1
    assert ds.removed_keys == 1
    assert ds.total_keys == 4


def test_score_mean_abs_delta_computed_correctly():
    diff = {
        "x": _td("changed", 0.0, 2.0),
        "y": _td("changed", 1.0, 3.0),
    }
    ds = compute_score(diff)
    assert abs(ds.mean_abs_delta - 2.0) < 1e-6
    assert abs(ds.max_abs_delta - 2.0) < 1e-6


def test_score_is_between_zero_and_one():
    diff = {"k": _td("changed", 0.0, 1e9)}
    ds = compute_score(diff)
    assert 0.0 <= ds.score <= 1.0


def test_empty_diff_returns_zero_score():
    ds = compute_score({})
    assert ds.score == 0.0
    assert ds.total_keys == 0


# ---------------------------------------------------------------------------
# format_score
# ---------------------------------------------------------------------------

def test_format_score_contains_score_label():
    ds = DiffScore(4, 1, 0, 0, 0.5, 0.5, 0.123)
    out = format_score(ds)
    assert "Diff Score" in out
    assert "0.1230" in out


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_diff_score_args(p)
    return p


def test_add_diff_score_args_registers_score():
    p = _make_parser()
    ns = p.parse_args([])
    assert hasattr(ns, "score")
    assert ns.score is False


def test_add_diff_score_args_registers_score_json():
    p = _make_parser()
    ns = p.parse_args([])
    assert hasattr(ns, "score_json")


def test_apply_diff_score_returns_none_when_flag_absent():
    p = _make_parser()
    ns = p.parse_args([])
    result = apply_diff_score(ns, {"w": _td("unchanged", 1.0, 1.0)})
    assert result is None


def test_apply_diff_score_returns_string_with_score_flag():
    p = _make_parser()
    ns = p.parse_args(["--score"])
    diff = {"a": _td("changed", 0.0, 1.0)}
    result = apply_diff_score(ns, diff)
    assert result is not None
    assert "Diff Score" in result


def test_apply_diff_score_json_returns_valid_json():
    p = _make_parser()
    ns = p.parse_args(["--score-json"])
    diff = {"a": _td("changed", 0.0, 1.0)}
    result = apply_diff_score(ns, diff)
    data = json.loads(result)
    assert "score" in data
    assert "changed_keys" in data
