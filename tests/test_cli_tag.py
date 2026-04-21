"""Tests for checkpoint_diff.cli_tag."""
import argparse
import pytest

from checkpoint_diff.tag import TagStore
from checkpoint_diff.cli_tag import add_tag_args, apply_tags
from checkpoint_diff.diff import CheckpointDiff, TensorDiff
import numpy as np


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_tag_args(p)
    return p


def _td() -> TensorDiff:
    a = np.array([1.0])
    return TensorDiff(status="changed", shape_a=a.shape, shape_b=a.shape,
                      mean_a=1.0, mean_b=2.0, std_a=0.0, std_b=0.0,
                      max_abs_diff=1.0)


def _make_diff() -> CheckpointDiff:
    return CheckpointDiff(
        added={}, removed={},
        changed={"layer.weight": _td(), "layer.bias": _td()},
        unchanged={},
    )


def test_add_tag_args_registers_tag():
    p = _make_parser()
    args = p.parse_args(["--tag", "layer.weight:frozen"])
    assert args.tag == ["layer.weight:frozen"]


def test_add_tag_args_filter_tag_default_none():
    p = _make_parser()
    args = p.parse_args([])
    assert args.filter_tag is None


def test_add_tag_args_show_tags_default_false():
    p = _make_parser()
    args = p.parse_args([])
    assert args.show_tags is False


def test_apply_tags_populates_store():
    p = _make_parser()
    args = p.parse_args(["--tag", "layer.weight:frozen"])
    diff, store = apply_tags(args, _make_diff())
    assert "frozen" in store.get("layer.weight")


def test_apply_tags_filter_reduces_diff():
    p = _make_parser()
    args = p.parse_args(["--tag", "layer.weight:frozen", "--filter-tag", "frozen"])
    diff, store = apply_tags(args, _make_diff())
    assert "layer.weight" in diff.changed
    assert "layer.bias" not in diff.changed


def test_apply_tags_invalid_format_raises():
    p = _make_parser()
    args = p.parse_args(["--tag", "badformat"])
    with pytest.raises(ValueError, match="KEY:TAG"):
        apply_tags(args, _make_diff())


def test_apply_tags_multiple_tags():
    p = _make_parser()
    args = p.parse_args(["--tag", "layer.weight:frozen", "--tag", "layer.bias:trainable"])
    _, store = apply_tags(args, _make_diff())
    assert store.get("layer.weight") == ["frozen"]
    assert store.get("layer.bias") == ["trainable"]
