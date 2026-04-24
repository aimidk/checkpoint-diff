"""Tests for checkpoint_diff.compare_sets."""
from __future__ import annotations

import numpy as np
import pytest

from checkpoint_diff.compare_sets import (
    CompareEntry,
    CompareSetResult,
    compare_against_reference,
    format_compare_set,
)


def _arr(*values):
    return np.array(values, dtype=np.float32)


@pytest.fixture()
def reference():
    return {"w1": _arr(1.0, 2.0), "w2": _arr(0.5, 0.5)}


@pytest.fixture()
def candidates(reference):
    identical = dict(reference)
    changed = {"w1": _arr(9.0, 9.0), "w2": _arr(0.5, 0.5)}
    very_different = {"w1": _arr(100.0, 100.0), "w2": _arr(50.0, 50.0)}
    return {"identical": identical, "changed": changed, "very_different": very_different}


def test_compare_returns_one_entry_per_candidate(reference, candidates):
    result = compare_against_reference(reference, candidates)
    assert len(result.entries) == 3


def test_reference_label_stored(reference, candidates):
    result = compare_against_reference(reference, candidates, reference_label="base")
    assert result.reference_label == "base"


def test_identical_candidate_has_zero_score(reference, candidates):
    result = compare_against_reference(reference, candidates)
    identical_entry = next(e for e in result.entries if e.label == "identical")
    assert identical_entry.score.overall == pytest.approx(0.0)


def test_ranked_orders_by_score_descending(reference, candidates):
    result = compare_against_reference(reference, candidates)
    ranked = result.ranked()
    scores = [e.score.overall for e in ranked]
    assert scores == sorted(scores, reverse=True)


def test_very_different_is_ranked_first(reference, candidates):
    result = compare_against_reference(reference, candidates)
    assert result.ranked()[0].label == "very_different"


def test_format_compare_set_contains_reference_label(reference, candidates):
    result = compare_against_reference(reference, candidates, reference_label="mybase")
    report = format_compare_set(result)
    assert "mybase" in report


def test_format_compare_set_contains_all_labels(reference, candidates):
    result = compare_against_reference(reference, candidates)
    report = format_compare_set(result)
    for label in candidates:
        assert label in report


def test_format_compare_set_top_n_limits_rows(reference, candidates):
    result = compare_against_reference(reference, candidates)
    report = format_compare_set(result, top_n=1)
    # Only the top entry should appear
    assert "very_different" in report
    assert "identical" not in report


def test_empty_candidates_produces_empty_entries(reference):
    result = compare_against_reference(reference, {})
    assert result.entries == []
    assert result.ranked() == []
