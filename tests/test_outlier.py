"""Tests for checkpoint_diff.outlier."""
import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff, CheckpointDiff
from checkpoint_diff.outlier import detect_outliers, format_outliers, OutlierResult


def _td(a, b=None, status="changed"):
    return TensorDiff(
        tensor_a=np.array(a, dtype=np.float32),
        tensor_b=np.array(b, dtype=np.float32) if b is not None else None,
        status=status,
    )


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(kwargs)


def test_no_outliers_when_within_limits():
    diff = _make_diff(w=_td([0.1, 0.2], [0.1, 0.2]))
    results = detect_outliers(diff, max_abs_mean=10.0, max_std=10.0)
    assert results == []


def test_detects_high_mean():
    diff = _make_diff(w=_td([100.0, 200.0], [100.0, 200.0]))
    results = detect_outliers(diff, max_abs_mean=50.0)
    assert len(results) == 1
    assert results[0].key == "w"
    assert "mean" in results[0].reason


def test_detects_high_std():
    diff = _make_diff(w=_td([0.0, 1000.0], [0.0, 1000.0]))
    results = detect_outliers(diff, max_std=1.0)
    assert len(results) == 1
    assert "std" in results[0].reason


def test_detects_high_abs_max():
    diff = _make_diff(w=_td([0.0, 999.0], [0.0, 999.0]))
    results = detect_outliers(diff, max_abs_max=100.0)
    assert len(results) == 1
    assert "abs_max" in results[0].reason


def test_unchanged_skipped_by_default():
    diff = _make_diff(w=_td([500.0, 500.0], [500.0, 500.0], status="unchanged"))
    results = detect_outliers(diff, max_abs_mean=1.0)
    assert results == []


def test_added_tensor_uses_tensor_b():
    diff = _make_diff(new=_td(None, [300.0, 300.0], status="added"))
    results = detect_outliers(diff, max_abs_mean=10.0)
    assert len(results) == 1


def test_format_outliers_no_results():
    assert format_outliers([]) == "No outliers detected."


def test_format_outliers_contains_key():
    r = OutlierResult(key="layer.weight", mean=200.0, std=5.0, abs_max=210.0, reason="|mean|=200 > 50")
    output = format_outliers([r])
    assert "layer.weight" in output
    assert "200" in output
