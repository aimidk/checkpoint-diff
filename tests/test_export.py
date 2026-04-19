"""Tests for checkpoint_diff.export."""
import csv
import io
import json

import numpy as np
import pytest

from checkpoint_diff.diff import compute_diff
from checkpoint_diff.export import export_diff, export_json, export_csv


def _make_checkpoints():
    a = {
        "layer1.weight": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "layer1.bias": np.array([0.5], dtype=np.float32),
        "only_in_a": np.array([9.0], dtype=np.float32),
    }
    b = {
        "layer1.weight": np.array([1.0, 2.5, 3.0], dtype=np.float32),
        "layer1.bias": np.array([0.5], dtype=np.float32),
        "only_in_b": np.array([7.0], dtype=np.float32),
    }
    return a, b


def test_export_json_returns_string():
    a, b = _make_checkpoints()
    diff = compute_diff(a, b)
    result = export_json(diff)
    assert isinstance(result, str)
    records = json.loads(result)
    assert isinstance(records, list)


def test_export_json_contains_expected_keys():
    a, b = _make_checkpoints()
    diff = compute_diff(a, b)
    records = json.loads(export_json(diff))
    keys = {r["key"] for r in records}
    assert "layer1.weight" in keys
    assert "only_in_a" in keys
    assert "only_in_b" in keys


def test_export_json_status_values():
    a, b = _make_checkpoints()
    diff = compute_diff(a, b)
    records = {r["key"]: r for r in json.loads(export_json(diff))}
    assert records["only_in_a"]["status"] == "removed"
    assert records["only_in_b"]["status"] == "added"
    assert records["layer1.bias"]["status"] == "unchanged"
    assert records["layer1.weight"]["status"] == "changed"


def test_export_csv_returns_string():
    a, b = _make_checkpoints()
    diff = compute_diff(a, b)
    result = export_csv(diff)
    assert isinstance(result, str)
    rows = list(csv.DictReader(io.StringIO(result)))
    assert len(rows) > 0


def test_export_csv_has_header():
    a, b = _make_checkpoints()
    diff = compute_diff(a, b)
    result = export_csv(diff)
    assert result.startswith("key,status")


def test_export_diff_json():
    a, b = _make_checkpoints()
    diff = compute_diff(a, b)
    out = export_diff(diff, "json")
    json.loads(out)  # must be valid JSON


def test_export_diff_csv():
    a, b = _make_checkpoints()
    diff = compute_diff(a, b)
    out = export_diff(diff, "csv")
    rows = list(csv.DictReader(io.StringIO(out)))
    assert len(rows) == len(diff)


def test_export_diff_unsupported_format():
    a, b = _make_checkpoints()
    diff = compute_diff(a, b)
    with pytest.raises(ValueError, match="Unsupported export format"):
        export_diff(diff, "xml")
