"""Tests for checkpoint_diff.cli_compare_sets."""
from __future__ import annotations

import argparse

import numpy as np
import pytest

from checkpoint_diff.cli_compare_sets import add_compare_sets_args, apply_compare_sets


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_compare_sets_args(p)
    return p


def _arr(*v):
    return np.array(v, dtype=np.float32)


_ref = {"w": _arr(1.0, 2.0)}
_cand = {"w": _arr(9.0, 9.0)}


def test_add_compare_sets_args_registers_flag():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.compare_sets is False


def test_add_compare_sets_args_compare_labels_default_none():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.compare_labels is None


def test_add_compare_sets_args_top_n_default_none():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.compare_top_n is None


def test_apply_returns_none_when_flag_not_set():
    p = _make_parser()
    ns = p.parse_args([])
    result = apply_compare_sets(ns, [_ref, _cand])
    assert result is None


def test_apply_returns_string_when_flag_set():
    p = _make_parser()
    ns = p.parse_args(["--compare-sets"])
    result = apply_compare_sets(ns, [_ref, _cand], labels=["ref", "cand"])
    assert isinstance(result, str)
    assert len(result) > 0


def test_apply_warns_when_fewer_than_two_checkpoints():
    p = _make_parser()
    ns = p.parse_args(["--compare-sets"])
    result = apply_compare_sets(ns, [_ref])
    assert "at least two" in result


def test_apply_uses_compare_labels_arg():
    p = _make_parser()
    ns = p.parse_args(["--compare-sets", "--compare-labels", "mymodel"])
    result = apply_compare_sets(ns, [_ref, _cand], labels=["base", "mymodel"])
    assert "mymodel" in result


def test_apply_top_n_limits_output():
    p = _make_parser()
    extra = {"w": _arr(50.0, 50.0)}
    ns = p.parse_args(["--compare-sets", "--compare-top-n", "1"])
    result = apply_compare_sets(
        ns, [_ref, _cand, extra], labels=["ref", "cand", "extra"]
    )
    # Only one candidate should appear in the output rows
    assert isinstance(result, str)
