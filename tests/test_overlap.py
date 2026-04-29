"""Tests for checkpoint_diff.overlap."""
from __future__ import annotations

import math

import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff
from checkpoint_diff.overlap import (
    OverlapResult,
    compute_overlap,
    format_overlap,
)


def _td(status: str) -> TensorDiff:
    arr = np.array([1.0, 2.0, 3.0])
    a = None if status == "added" else arr
    b = None if status == "removed" else arr
    return TensorDiff(key="k", status=status, tensor_a=a, tensor_b=b)


def _make_diff(statuses: list[str]) -> dict:
    result = {}
    for i, s in enumerate(statuses):
        td = _td(s)
        result[f"key_{i}"] = td
    return result


def test_all_shared_keys_jaccard_is_one():
    diff = _make_diff(["changed", "unchanged", "unchanged"])
    result = compute_overlap(diff)
    assert result.jaccard == pytest.approx(1.0)
    assert result.dice == pytest.approx(1.0)


def test_no_shared_keys_jaccard_is_zero():
    diff = _make_diff(["removed", "removed", "added", "added"])
    result = compute_overlap(diff)
    assert result.jaccard == pytest.approx(0.0)
    assert result.dice == pytest.approx(0.0)


def test_keys_only_in_a_counted_correctly():
    diff = _make_diff(["removed", "removed", "unchanged"])
    result = compute_overlap(diff)
    assert result.keys_only_in_a == 2
    assert result.keys_only_in_b == 0
    assert result.keys_in_both == 1


def test_keys_only_in_b_counted_correctly():
    diff = _make_diff(["added", "unchanged"])
    result = compute_overlap(diff)
    assert result.keys_only_in_b == 1
    assert result.keys_only_in_a == 0
    assert result.keys_in_both == 1


def test_total_keys_is_sum_of_all():
    diff = _make_diff(["added", "removed", "changed", "unchanged"])
    result = compute_overlap(diff)
    assert result.total_keys == 4


def test_empty_diff_returns_nan():
    result = compute_overlap({})
    assert math.isnan(result.jaccard)
    assert math.isnan(result.dice)
    assert result.total_keys == 0


def test_jaccard_partial_overlap():
    # 2 shared, 1 only in A, 1 only in B  =>  jaccard = 2/4 = 0.5
    diff = _make_diff(["unchanged", "unchanged", "removed", "added"])
    result = compute_overlap(diff)
    assert result.jaccard == pytest.approx(0.5)


def test_dice_partial_overlap():
    # 2 shared, total_a=3, total_b=3  =>  dice = 4/6 ≈ 0.6667
    diff = _make_diff(["unchanged", "unchanged", "removed", "added"])
    result = compute_overlap(diff)
    assert result.dice == pytest.approx(4 / 6)


def test_format_overlap_contains_jaccard():
    diff = _make_diff(["unchanged"])
    result = compute_overlap(diff)
    text = format_overlap(result)
    assert "Jaccard" in text
    assert "Dice" in text


def test_format_overlap_shows_counts():
    diff = _make_diff(["removed", "added", "changed"])
    result = compute_overlap(diff)
    text = format_overlap(result)
    assert str(result.keys_only_in_a) in text
    assert str(result.keys_only_in_b) in text
