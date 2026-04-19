"""Tests for checkpoint_diff.filter."""

from __future__ import annotations

import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff, compute_diff
from checkpoint_diff.filter import (
    filter_by_key_pattern,
    filter_by_max_abs_mean,
    filter_by_status,
)


def _make_diff() -> dict:
    a = {
        "layer.weight": np.array([1.0, 2.0]),
        "layer.bias": np.array([0.5]),
        "only_a": np.array([9.0]),
    }
    b = {
        "layer.weight": np.array([1.0, 2.5]),
        "layer.bias": np.array([0.5]),
        "only_b": np.array([3.0]),
    }
    return compute_diff(a, b)


def test_filter_unchanged_excluded():
    diff = _make_diff()
    result = filter_by_status(diff, include_unchanged=False)
    assert "layer.bias" not in result
    assert "layer.weight" in result


def test_filter_changed_excluded():
    diff = _make_diff()
    result = filter_by_status(diff, include_changed=False)
    assert "layer.weight" not in result
    assert "layer.bias" in result


def test_filter_added_removed_excluded():
    diff = _make_diff()
    result = filter_by_status(diff, include_added=False, include_removed=False)
    assert "only_b" not in result
    assert "only_a" not in result
    assert "layer.bias" in result


def test_filter_by_key_pattern_single():
    diff = _make_diff()
    result = filter_by_key_pattern(diff, ["layer.*"])
    assert set(result.keys()) == {"layer.weight", "layer.bias"}


def test_filter_by_key_pattern_empty_returns_all():
    diff = _make_diff()
    result = filter_by_key_pattern(diff, [])
    assert set(result.keys()) == set(diff.keys())


def test_filter_by_max_abs_mean_keeps_large_diffs():
    diff = _make_diff()
    # layer.weight mean_diff should be 0.25; threshold below that keeps it
    result = filter_by_max_abs_mean(diff, threshold=0.1)
    assert "layer.weight" in result


def test_filter_by_max_abs_mean_removes_small_diffs():
    diff = _make_diff()
    # threshold above 0.25 should remove layer.weight
    result = filter_by_max_abs_mean(diff, threshold=1.0)
    assert "layer.weight" not in result
    # non-changed entries survive regardless
    assert "layer.bias" in result
    assert "only_a" in result
