"""Tests for checkpoint_diff.similarity."""
import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff, CheckpointDiff
from checkpoint_diff.similarity import (
    compute_similarity,
    format_similarity,
    TensorSimilarity,
)


def _td(a, b, status="changed"):
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    return TensorDiff(status=status, array_a=a_arr, array_b=b_arr)


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(kwargs)


def test_cosine_identical_vectors():
    diff = _make_diff(w=_td([1.0, 0.0], [1.0, 0.0]))
    report = compute_similarity(diff)
    assert abs(report["w"].cosine - 1.0) < 1e-6


def test_cosine_orthogonal_vectors():
    diff = _make_diff(w=_td([1.0, 0.0], [0.0, 1.0]))
    report = compute_similarity(diff)
    assert abs(report["w"].cosine) < 1e-6


def test_cosine_opposite_vectors():
    diff = _make_diff(w=_td([1.0, 0.0], [-1.0, 0.0]))
    report = compute_similarity(diff)
    assert abs(report["w"].cosine + 1.0) < 1e-6


def test_l2_norm_delta():
    diff = _make_diff(w=_td([3.0, 4.0], [0.0, 0.0]))
    report = compute_similarity(diff)
    assert abs(report["w"].l2_norm_a - 5.0) < 1e-6
    assert abs(report["w"].l2_norm_b) < 1e-6
    assert abs(report["w"].l2_norm_delta - 5.0) < 1e-6


def test_added_key_excluded():
    td = TensorDiff(status="added", array_a=None, array_b=np.array([1.0]))
    diff = _make_diff(w=td)
    report = compute_similarity(diff)
    assert "w" not in report


def test_removed_key_excluded():
    td = TensorDiff(status="removed", array_a=np.array([1.0]), array_b=None)
    diff = _make_diff(w=td)
    report = compute_similarity(diff)
    assert "w" not in report


def test_zero_norm_cosine_is_none():
    diff = _make_diff(w=_td([0.0, 0.0], [1.0, 0.0]))
    report = compute_similarity(diff)
    assert report["w"].cosine is None


def test_format_similarity_no_report():
    out = format_similarity({})
    assert "No changed" in out


def test_format_similarity_contains_key():
    diff = _make_diff(layer=_td([1.0], [2.0]))
    report = compute_similarity(diff)
    out = format_similarity(report)
    assert "layer" in out
