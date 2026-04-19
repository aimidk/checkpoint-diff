"""Tests for checkpoint_diff.diff and checkpoint_diff.report."""

import numpy as np
import pytest

from checkpoint_diff.diff import compute_diff, CheckpointDiff
from checkpoint_diff.report import format_report


CKPT_A = {
    "layer1.weight": np.array([[1.0, 2.0], [3.0, 4.0]]),
    "layer1.bias": np.array([0.1, 0.2]),
    "layer2.weight": np.array([5.0, 6.0]),
}

CKPT_B = {
    "layer1.weight": np.array([[1.0, 2.0], [3.0, 4.0]]),  # unchanged
    "layer1.bias": np.array([0.1, 0.9]),                   # changed
    "layer3.weight": np.array([7.0, 8.0]),                 # added
    # layer2.weight removed
}


def test_added_keys():
    diff = compute_diff(CKPT_A, CKPT_B)
    assert diff.added == ["layer3.weight"]


def test_removed_keys():
    diff = compute_diff(CKPT_A, CKPT_B)
    assert diff.removed == ["layer2.weight"]


def test_unchanged_keys():
    diff = compute_diff(CKPT_A, CKPT_B)
    assert "layer1.weight" in diff.unchanged


def test_changed_tensor_stats():
    diff = compute_diff(CKPT_A, CKPT_B)
    assert len(diff.changed) == 1
    td = diff.changed[0]
    assert td.key == "layer1.bias"
    assert td.max_abs_diff == pytest.approx(0.7, abs=1e-6)
    assert td.mean_abs_diff == pytest.approx(0.35, abs=1e-6)


def test_shape_mismatch():
    a = {"w": np.ones((3, 4))}
    b = {"w": np.ones((4, 3))}
    diff = compute_diff(a, b)
    assert len(diff.changed) == 1
    assert diff.changed[0].max_abs_diff is None


def test_identical_checkpoints():
    diff = compute_diff(CKPT_A, CKPT_A)
    assert not diff.has_differences


def test_format_report_identical():
    diff = compute_diff(CKPT_A, CKPT_A)
    report = format_report(diff)
    assert "identical" in report.lower()


def test_format_report_contains_key_names():
    diff = compute_diff(CKPT_A, CKPT_B)
    report = format_report(diff)
    assert "layer3.weight" in report
    assert "layer2.weight" in report
    assert "layer1.bias" in report


def test_format_report_verbose_shows_unchanged():
    diff = compute_diff(CKPT_A, CKPT_B)
    report = format_report(diff, verbose=True)
    assert "layer1.weight" in report
