"""Tests for checkpoint_diff.group."""
from __future__ import annotations

import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff
from checkpoint_diff.group import (
    GroupResult,
    format_groups,
    group_by_map,
    group_by_prefix,
)


def _td(status: str = "changed") -> TensorDiff:
    a = np.array([1.0, 2.0])
    b = np.array([1.5, 2.5])
    if status == "added":
        return TensorDiff(key="k", status="added", tensor_a=None, tensor_b=b)
    if status == "removed":
        return TensorDiff(key="k", status="removed", tensor_a=a, tensor_b=None)
    return TensorDiff(key="k", status=status, tensor_a=a, tensor_b=b)


def _make_diff():
    return {
        "encoder.weight": _td("changed"),
        "encoder.bias": _td("unchanged"),
        "decoder.weight": _td("added"),
        "decoder.bias": _td("removed"),
        "head": _td("changed"),
    }


def test_group_by_prefix_creates_correct_groups():
    diff = _make_diff()
    groups = group_by_prefix(diff)
    assert set(groups.keys()) == {"encoder", "decoder", "head"}


def test_group_by_prefix_key_counts():
    diff = _make_diff()
    groups = group_by_prefix(diff)
    assert len(groups["encoder"].keys) == 2
    assert len(groups["decoder"].keys) == 2
    assert len(groups["head"].keys) == 1


def test_group_by_prefix_status_counts():
    diff = _make_diff()
    groups = group_by_prefix(diff)
    assert groups["encoder"].num_changed == 1
    assert groups["decoder"].num_added == 1
    assert groups["decoder"].num_removed == 1


def test_group_by_prefix_custom_sep():
    diff = {"encoder/weight": _td(), "encoder/bias": _td(), "head": _td()}
    groups = group_by_prefix(diff, sep="/")
    assert "encoder" in groups
    assert len(groups["encoder"].keys) == 2


def test_group_by_map_assigns_correct_groups():
    diff = _make_diff()
    mapping = {
        "encoder.weight": "backbone",
        "encoder.bias": "backbone",
        "decoder.weight": "head_block",
    }
    groups = group_by_map(diff, mapping)
    assert "backbone" in groups
    assert "head_block" in groups
    assert "other" in groups  # unmapped keys


def test_group_by_map_default_group():
    diff = {"a.b": _td(), "x.y": _td()}
    groups = group_by_map(diff, {"a.b": "grp1"}, default_group="misc")
    assert "misc" in groups
    assert "x.y" in groups["misc"].keys


def test_format_groups_contains_group_name():
    diff = _make_diff()
    groups = group_by_prefix(diff)
    output = format_groups(groups)
    assert "encoder" in output
    assert "decoder" in output


def test_group_result_counts_are_correct():
    gr = GroupResult(group_name="test")
    gr.diffs["a"] = _td("changed")
    gr.diffs["b"] = _td("added")
    gr.diffs["c"] = _td("removed")
    gr.diffs["d"] = _td("unchanged")
    assert gr.num_changed == 1
    assert gr.num_added == 1
    assert gr.num_removed == 1
