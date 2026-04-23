"""Tests for checkpoint_diff.drift and checkpoint_diff.cli_drift."""
from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.drift import DriftReport, detect_drift, format_drift
from checkpoint_diff.cli_drift import add_drift_args, apply_drift


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _td(status: str, a=None, b=None) -> TensorDiff:
    from checkpoint_diff.stats import _compute_stats  # reuse existing helper

    stats_a = _compute_stats(a) if a is not None else None
    stats_b = _compute_stats(b) if b is not None else None
    return TensorDiff(status=status, stats_a=stats_a, stats_b=stats_b)


def _make_diff(entries: Dict[str, TensorDiff]) -> CheckpointDiff:
    return CheckpointDiff(entries)


# ---------------------------------------------------------------------------
# detect_drift
# ---------------------------------------------------------------------------

def test_no_drift_when_identical():
    a = np.ones(100)
    diff = _make_diff({"w": _td("changed", a, a.copy())})
    report = detect_drift(diff, mean_threshold=0.1, std_threshold=0.1)
    assert report.flagged == []


def test_flags_large_mean_shift():
    a = np.zeros(50) + 1.0
    b = np.zeros(50) + 2.0  # 100 % relative shift
    diff = _make_diff({"layer.weight": _td("changed", a, b)})
    report = detect_drift(diff, mean_threshold=0.1)
    assert len(report.flagged) == 1
    assert report.flagged[0].key == "layer.weight"


def test_flags_large_std_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 200)
    b = rng.normal(0, 5, 200)  # std roughly 5× larger
    diff = _make_diff({"bias": _td("changed", a, b)})
    report = detect_drift(diff, std_threshold=0.1)
    assert any(r.key == "bias" for r in report.flagged)


def test_added_removed_keys_skipped():
    a = np.ones(10)
    diff = _make_diff({
        "added_key": _td("added", None, a),
        "removed_key": _td("removed", a, None),
    })
    report = detect_drift(diff)
    assert report.results == []


def test_include_unchanged_flag():
    a = np.ones(50)
    diff = _make_diff({"stable": _td("unchanged", a, a.copy())})
    report = detect_drift(diff, include_unchanged=True)
    assert len(report.results) == 1
    assert not report.results[0].flagged


def test_results_sorted_by_drift_descending():
    a = np.ones(50)
    big_shift = np.ones(50) * 10.0
    small_shift = np.ones(50) * 1.1
    diff = _make_diff({
        "small": _td("changed", a, small_shift),
        "big": _td("changed", a, big_shift),
    })
    report = detect_drift(diff, mean_threshold=0.0)
    assert report.results[0].key == "big"


# ---------------------------------------------------------------------------
# format_drift
# ---------------------------------------------------------------------------

def test_format_drift_no_results_returns_message():
    report = DriftReport(results=[])
    assert "No drift" in format_drift(report)


def test_format_drift_contains_key_name():
    a = np.ones(50)
    b = np.ones(50) * 3.0
    diff = _make_diff({"encoder.fc": _td("changed", a, b)})
    report = detect_drift(diff, mean_threshold=0.0)
    text = format_drift(report)
    assert "encoder.fc" in text


def test_format_drift_top_n_limits_rows():
    a = np.ones(20)
    entries = {f"k{i}": _td("changed", a, a * (2 + i)) for i in range(5)}
    diff = _make_diff(entries)
    report = detect_drift(diff, mean_threshold=0.0)
    text = format_drift(report, top_n=2)
    # Only 2 data rows (plus header + separator)
    data_lines = [l for l in text.splitlines() if l.startswith("k")]
    assert len(data_lines) == 2


# ---------------------------------------------------------------------------
# cli_drift
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_drift_args(p)
    return p


def test_add_drift_args_registers_flag():
    p = _make_parser()
    args = p.parse_args(["--drift"])
    assert args.drift is True


def test_add_drift_args_defaults():
    p = _make_parser()
    args = p.parse_args([])
    assert args.drift is False
    assert args.drift_mean_threshold == pytest.approx(0.1)
    assert args.drift_std_threshold == pytest.approx(0.1)
    assert args.drift_top is None
    assert args.drift_include_unchanged is False


def test_apply_drift_returns_none_when_flag_absent():
    p = _make_parser()
    args = p.parse_args([])
    diff = _make_diff({})
    assert apply_drift(args, diff) is None


def test_apply_drift_returns_string_when_flag_set():
    a = np.ones(50)
    b = np.ones(50) * 5.0
    diff = _make_diff({"w": _td("changed", a, b)})
    p = _make_parser()
    args = p.parse_args(["--drift", "--drift-mean-threshold", "0.0"])
    result = apply_drift(args, diff)
    assert isinstance(result, str)
    assert "w" in result
