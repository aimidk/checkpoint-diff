"""Tests for checkpoint_diff.align."""
import numpy as np
import pytest

from checkpoint_diff.align import (
    align_checkpoints,
    auto_detect_prefix,
    strip_prefix,
)


def _arr(*shape):
    return np.zeros(shape)


def test_strip_prefix_removes_matching():
    keys = ["model.weight", "model.bias", "other"]
    result = strip_prefix(keys, "model.")
    assert "weight" in result
    assert "bias" in result
    assert "other" in result
    assert result["weight"] == "model.weight"


def test_strip_prefix_no_match_unchanged():
    keys = ["layer.weight"]
    result = strip_prefix(keys, "model.")
    assert result["layer.weight"] == "layer.weight"


def test_auto_detect_prefix_finds_common():
    keys_a = ["module.layer.weight", "module.layer.bias"]
    keys_b = ["layer.weight", "layer.bias"]
    pa, pb = auto_detect_prefix(keys_a, keys_b)
    assert pa == "module."
    assert pb == ""


def test_auto_detect_prefix_empty_keys():
    pa, pb = auto_detect_prefix([], [])
    assert pa == ""
    assert pb == ""


def test_align_checkpoints_strips_prefixes():
    ckpt_a = {"module.w": _arr(3), "module.b": _arr(3)}
    ckpt_b = {"w": _arr(3), "b": _arr(3)}
    a, b = align_checkpoints(ckpt_a, ckpt_b, prefix_a="module.", prefix_b="")
    assert set(a.keys()) == {"w", "b"}
    assert set(b.keys()) == {"w", "b"}


def test_align_checkpoints_auto():
    ckpt_a = {"net.w": _arr(4), "net.b": _arr(4)}
    ckpt_b = {"w": _arr(4), "b": _arr(4)}
    a, b = align_checkpoints(ckpt_a, ckpt_b, auto_align=True)
    assert set(a.keys()) == set(b.keys())


def test_align_checkpoints_no_prefix_unchanged():
    ckpt_a = {"w": _arr(2)}
    ckpt_b = {"w": _arr(2)}
    a, b = align_checkpoints(ckpt_a, ckpt_b)
    assert list(a.keys()) == ["w"]
    assert list(b.keys()) == ["w"]
