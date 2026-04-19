"""Tests for cli_align helpers."""
import argparse

import numpy as np
import pytest

from checkpoint_diff.cli_align import add_align_args, apply_alignment


def _make_parser():
    p = argparse.ArgumentParser()
    add_align_args(p)
    return p


def test_add_align_args_registers_prefix_a():
    p = _make_parser()
    ns = p.parse_args(["--prefix-a", "module."])
    assert ns.prefix_a == "module."


def test_add_align_args_registers_prefix_b():
    p = _make_parser()
    ns = p.parse_args(["--prefix-b", "net."])
    assert ns.prefix_b == "net."


def test_add_align_args_auto_align_flag():
    p = _make_parser()
    ns = p.parse_args(["--auto-align"])
    assert ns.auto_align is True


def test_add_align_args_defaults():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.prefix_a == ""
    assert ns.prefix_b == ""
    assert ns.auto_align is False


def test_apply_alignment_strips_correctly():
    p = _make_parser()
    ns = p.parse_args(["--prefix-a", "old."])
    ckpt_a = {"old.weight": np.zeros(3)}
    ckpt_b = {"weight": np.zeros(3)}
    a, b = apply_alignment(ns, ckpt_a, ckpt_b)
    assert "weight" in a
    assert "weight" in b


def test_apply_alignment_auto():
    p = _make_parser()
    ns = p.parse_args(["--auto-align"])
    ckpt_a = {"m.w": np.ones(2), "m.b": np.ones(2)}
    ckpt_b = {"w": np.ones(2), "b": np.ones(2)}
    a, b = apply_alignment(ns, ckpt_a, ckpt_b)
    assert set(a.keys()) == set(b.keys())
