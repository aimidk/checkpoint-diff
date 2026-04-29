"""Tests for checkpoint_diff.dead_neurons."""
import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.dead_neurons import (
    DeadNeuronRow,
    _dead_count,
    compute_dead_neurons,
    format_dead_neurons,
)


def _td(a, b, status="changed"):
    return TensorDiff(
        status=status,
        array_a=np.array(a, dtype=float) if a is not None else None,
        array_b=np.array(b, dtype=float) if b is not None else None,
        mean_a=float(np.mean(a)) if a is not None else None,
        mean_b=float(np.mean(b)) if b is not None else None,
        std_a=float(np.std(a)) if a is not None else None,
        std_b=float(np.std(b)) if b is not None else None,
        shape_a=np.array(a).shape if a is not None else None,
        shape_b=np.array(b).shape if b is not None else None,
    )


def _make_diff(entries: dict) -> CheckpointDiff:
    return CheckpointDiff(tensors=entries)


# --- _dead_count ---

def test_dead_count_all_zeros():
    total, dead = _dead_count(np.zeros(10), eps=1e-6)
    assert total == 10
    assert dead == 10


def test_dead_count_no_zeros():
    total, dead = _dead_count(np.ones(5), eps=1e-6)
    assert total == 5
    assert dead == 0


def test_dead_count_none_returns_zeros():
    assert _dead_count(None, eps=1e-6) == (0, 0)


def test_dead_count_mixed():
    arr = np.array([0.0, 1.0, 0.0, 2.0])
    total, dead = _dead_count(arr, eps=1e-6)
    assert total == 4
    assert dead == 2


def test_dead_count_respects_eps():
    """Values within eps of zero should be counted as dead."""
    arr = np.array([0.0, 5e-7, -5e-7, 1.0])
    total, dead = _dead_count(arr, eps=1e-6)
    assert total == 4
    assert dead == 3


# --- compute_dead_neurons ---

def test_compute_dead_neurons_basic():
    diff = _make_diff({
        "w": _td([0.0, 1.0, 0.0], [0.0, 0.0, 0.0]),
    })
    rows = compute_dead_neurons(diff)
    assert len(rows) == 1
    r = rows[0]
    assert r.key == "w"
    assert r.dead_a == 2
    assert r.dead_b == 3
    assert r.dead_delta > 0


def test_compute_dead_neurons_skips_removed():
    diff = _make_diff({
        "gone": _td([1.0, 2.0], None, status="removed"),
        "kept": _td([0.0], [0.0]),
    })
    rows = compute_dead_neurons(diff)
    keys = [r.key for r in rows]
    assert "gone" not in keys
    assert "kept" in keys


def test_compute_dead_neurons_only_changed_filters_stable():
    diff = _make_diff({
        "stable": _td([0.0, 0.0], [0.0, 0.0]),
        "moving": _td([0.0, 1.0], [0.0, 0.0]),
    })
    rows = compute_dead_neurons(diff, only_changed=True)
    keys = [r.key for r in rows]
    assert "stable" not in keys
    assert "moving" in keys


def test_compute_dead_neurons_sorted_by_abs_delta():
    diff = _make_diff({
        "small": _td([0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]),
        "large": _td([1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]),
    })
    rows = compute_dead_neurons(diff)
    assert rows[0].key == "large"


def test_compute_dead_neurons_empty_diff():
    """An empty diff should return an empty list without errors."""
    diff = _make_diff({})
    rows = compute_dead_neurons(diff)
    assert rows == []


# --- format_dead_neurons ---

def test_format_dead_neurons_contains_key():
    diff = _make_diff({"layer.weight": _td([0.0, 1.0], [0.0, 0.0])})
    rows = compute_dead_neurons(diff)
    out = format_dead_neurons(rows)
    assert "layer.weight" in out


def test_format_dead_neurons_empty_rows():
    """Formatting an empty row list should return a non-error string."""
    out = format_dead_neurons([])
    assert isinstance(out, str)
