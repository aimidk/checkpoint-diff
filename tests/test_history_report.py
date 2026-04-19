"""Tests for checkpoint_diff.history_report."""
import csv
import io

import pytest

from checkpoint_diff.history import KeyTrend
from checkpoint_diff.history_report import (
    export_trends_csv,
    format_trend_table,
    print_trend_report,
)


def _trend(key: str, deltas: list) -> KeyTrend:
    return KeyTrend(
        key=key,
        deltas=deltas,
        total_delta=sum(deltas),
        max_delta=max(abs(d) for d in deltas) if deltas else 0.0,
        num_steps=len(deltas),
    )


def test_format_trend_table_contains_key():
    trends = [_trend("layer.weight", [0.1, 0.2])]
    table = format_trend_table(trends)
    assert "layer.weight" in table


def test_format_trend_table_sorted_by_abs_delta():
    trends = [
        _trend("small", [0.01]),
        _trend("large", [5.0]),
    ]
    table = format_trend_table(trends)
    assert table.index("large") < table.index("small")


def test_format_trend_table_empty():
    assert "No trends" in format_trend_table([])


def test_format_trend_table_top_n_limits_rows():
    trends = [_trend(f"key_{i}", [float(i)]) for i in range(1, 20)]
    table = format_trend_table(trends, top_n=3)
    # header + sep + 3 data rows
    data_lines = [l for l in table.strip().splitlines() if l and not l.startswith("-") and "Key" not in l]
    assert len(data_lines) == 3


def test_export_trends_csv_headers():
    csv_str = export_trends_csv([_trend("a.b", [0.5, -0.1])])
    reader = csv.DictReader(io.StringIO(csv_str))
    rows = list(reader)
    assert rows[0]["key"] == "a.b"
    assert "total_delta" in rows[0]
    assert "deltas" in rows[0]


def test_export_trends_csv_delta_values():
    csv_str = export_trends_csv([_trend("x", [1.0, 2.0])])
    reader = csv.DictReader(io.StringIO(csv_str))
    row = next(reader)
    assert row["num_steps"] == "2"
    parts = row["deltas"].split("|")
    assert len(parts) == 2


def test_print_trend_report_runs(capsys):
    trends = [_trend("w", [0.3])]
    print_trend_report(trends)
    captured = capsys.readouterr()
    assert "w" in captured.out
