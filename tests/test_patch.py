"""Tests for checkpoint_diff.patch."""
from __future__ import annotations

import numpy as np
import pytest

from checkpoint_diff.diff import compute_diff
from checkpoint_diff.patch import apply_patch, patch_summary


def _arr(*shape, fill=1.0):
    return np.full(shape, fill, dtype=np.float32)


@pytest.fixture()
def base():
    return {"w": _arr(3, fill=1.0), "b": _arr(3, fill=0.0)}


@pytest.fixture()
def other():
    return {"w": _arr(3, fill=2.0), "extra": _arr(3, fill=5.0)}


def test_apply_patch_changed_key_uses_other(base, other):
    diff = compute_diff(base, other)
    result = apply_patch(base, diff)
    np.testing.assert_array_equal(result["w"], other["w"])


def test_apply_patch_added_key_included_by_default(base, other):
    diff = compute_diff(base, other)
    result = apply_patch(base, diff)
    assert "extra" in result
    np.testing.assert_array_equal(result["extra"], other["extra"])


def test_apply_patch_added_key_excluded_when_skip_added(base, other):
    diff = compute_diff(base, other)
    result = apply_patch(base, diff, skip_added=True)
    assert "extra" not in result


def test_apply_patch_removed_key_absent_by_default(base, other):
    diff = compute_diff(base, other)
    result = apply_patch(base, diff)
    assert "b" not in result


def test_apply_patch_removed_key_kept_when_skip_removed(base, other):
    diff = compute_diff(base, other)
    result = apply_patch(base, diff, skip_removed=True)
    assert "b" in result
    np.testing.assert_array_equal(result["b"], base["b"])


def test_apply_patch_unchanged_key_preserved(base, other):
    base["shared"] = _arr(4, fill=9.0)
    other["shared"] = _arr(4, fill=9.0)
    diff = compute_diff(base, other)
    result = apply_patch(base, diff)
    np.testing.assert_array_equal(result["shared"], base["shared"])


def test_patch_summary_contains_counts(base, other):
    diff = compute_diff(base, other)
    summary = patch_summary(diff)
    assert "changed" in summary
    assert "added" in summary
    assert "removed" in summary


def test_patch_summary_starts_with_patch(base, other):
    diff = compute_diff(base, other)
    assert patch_summary(diff).startswith("Patch:")
