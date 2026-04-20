"""Tests for checkpoint_diff.annotation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from checkpoint_diff.annotation import (
    AnnotationStore,
    annotate_report,
    load_annotations,
    save_annotations,
)
from checkpoint_diff.diff import TensorDiff


def _td(status: str = "changed") -> TensorDiff:
    a = np.array([1.0, 2.0])
    b = np.array([1.5, 2.5])
    return TensorDiff(key="w", status=status, shape_a=a.shape, shape_b=b.shape,
                      array_a=a, array_b=b)


def _make_diff(*keys_statuses):
    """Return a plain dict acting as a minimal CheckpointDiff."""
    result = {}
    for key, status in keys_statuses:
        a = np.ones(3)
        result[key] = TensorDiff(key=key, status=status, shape_a=a.shape,
                                 shape_b=a.shape, array_a=a, array_b=a)
    return result


class TestAnnotationStore:
    def test_add_and_get(self):
        store = AnnotationStore()
        store.add("layer.weight", "suspicious spike")
        assert store.get("layer.weight") == "suspicious spike"

    def test_get_missing_returns_none(self):
        store = AnnotationStore()
        assert store.get("nonexistent") is None

    def test_remove_existing_returns_true(self):
        store = AnnotationStore()
        store.add("k", "note")
        assert store.remove("k") is True
        assert store.get("k") is None

    def test_remove_missing_returns_false(self):
        store = AnnotationStore()
        assert store.remove("ghost") is False


def test_save_and_load_roundtrip(tmp_path):
    store = AnnotationStore()
    store.add("fc.weight", "large shift")
    store.add("fc.bias", "minor")
    dest = tmp_path / "ann.json"
    save_annotations(store, dest)
    loaded = load_annotations(dest)
    assert loaded.notes == store.notes


def test_load_annotations_missing_notes_key(tmp_path):
    dest = tmp_path / "empty.json"
    dest.write_text(json.dumps({}))
    store = load_annotations(dest)
    assert store.notes == {}


def test_annotate_report_includes_note():
    diff = _make_diff(("layer.weight", "changed"), ("layer.bias", "unchanged"))
    store = AnnotationStore()
    store.add("layer.weight", "big delta")
    report = annotate_report(diff, store)
    assert "layer.weight" in report
    assert "big delta" in report


def test_annotate_report_no_note_for_key():
    diff = _make_diff(("a", "added"),)
    store = AnnotationStore()
    report = annotate_report(diff, store)
    assert "#" not in report


def test_annotate_report_sorted():
    diff = _make_diff(("z.w", "changed"), ("a.w", "changed"))
    store = AnnotationStore()
    report = annotate_report(diff, store)
    lines = report.splitlines()
    assert lines[0].startswith("a.w")
    assert lines[1].startswith("z.w")
