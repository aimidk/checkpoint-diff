"""Tests for checkpoint_diff.regression."""
from __future__ import annotations

import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.regression import (
    RegressionReport,
    RegressionResult,
    detect_regression,
    format_regression,
)


def _td(
    status: str = "changed",
    mean_a: float = 0.0,
    mean_b: float = 0.0,
) -> TensorDiff:
    return TensorDiff(
        status=status,
        shape_a=(4,),
        shape_b=(4,),
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=1.0,
        std_b=1.0,
    )


def _make_diff(**kwargs: TensorDiff) -> CheckpointDiff:
    return CheckpointDiff(tensors=kwargs)


def _ref(**kwargs: float) -> dict:
    return {k: np.array([v]) for k, v in kwargs.items()}


# ---------------------------------------------------------------------------

def test_no_regression_when_b_closer_to_ref():
    diff = _make_diff(w=_td("changed", mean_a=5.0, mean_b=1.0))
    ref = _ref(w=0.0)  # ref mean = 0; A is 5 away, B is 1 away
    report = detect_regression(diff, ref)
    assert len(report.flagged) == 0


def test_regression_when_b_further_from_ref():
    diff = _make_diff(w=_td("changed", mean_a=1.0, mean_b=5.0))
    ref = _ref(w=0.0)  # ref mean = 0; A is 1 away, B is 5 away
    report = detect_regression(diff, ref)
    assert len(report.flagged) == 1
    assert report.flagged[0].key == "w"
    assert report.flagged[0].direction == "away"


def test_direction_toward_when_b_closer():
    diff = _make_diff(w=_td("changed", mean_a=4.0, mean_b=1.0))
    ref = _ref(w=0.0)
    report = detect_regression(diff, ref)
    result = report.results[0]
    assert result.direction == "toward"
    assert not result.regressed


def test_tolerance_prevents_flagging_small_increase():
    diff = _make_diff(w=_td("changed", mean_a=1.0, mean_b=1.05))
    ref = _ref(w=0.0)
    report = detect_regression(diff, ref, tolerance=0.1)
    assert len(report.flagged) == 0


def test_key_missing_from_reference_is_skipped():
    diff = _make_diff(w=_td("changed", mean_a=1.0, mean_b=5.0))
    ref = {}  # no reference for 'w'
    report = detect_regression(diff, ref)
    assert report.results == []


def test_unchanged_tensors_skipped():
    diff = _make_diff(w=_td("unchanged", mean_a=1.0, mean_b=5.0))
    ref = _ref(w=0.0)
    report = detect_regression(diff, ref)
    assert report.results == []


def test_format_regression_no_regressions():
    report = RegressionReport(results=[])
    out = format_regression(report)
    assert "No regressions" in out


def test_format_regression_shows_flagged_key():
    diff = _make_diff(w=_td("changed", mean_a=1.0, mean_b=5.0))
    ref = _ref(w=0.0)
    report = detect_regression(diff, ref)
    out = format_regression(report)
    assert "w" in out
    assert "away" in out


def test_format_regression_show_all_includes_non_regressions():
    diff = _make_diff(
        w=_td("changed", mean_a=5.0, mean_b=1.0),  # toward
        b=_td("changed", mean_a=1.0, mean_b=5.0),  # away
    )
    ref = _ref(w=0.0, b=0.0)
    report = detect_regression(diff, ref)
    out = format_regression(report, show_all=True)
    assert "w" in out
    assert "b" in out
    assert "toward" in out
    assert "away" in out


def test_regression_report_flagged_property():
    """RegressionReport.flagged should only contain results where regressed=True."""
    results = [
        RegressionResult(key="a", direction="away", regressed=True, dist_a=1.0, dist_b=5.0),
        RegressionResult(key="b", direction="toward", regressed=False, dist_a=4.0, dist_b=1.0),
    ]
    report = RegressionReport(results=results)
    assert len(report.flagged) == 1
    assert report.flagged[0].key == "a"
