"""Tests for checkpoint_diff.sparkline."""

from __future__ import annotations

import pytest

from checkpoint_diff.sparkline import (
    SparklineRow,
    _normalize,
    render_sparkline,
    build_sparklines,
    format_sparklines,
)


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

def test_normalize_flat_sequence_returns_zeros():
    assert _normalize([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]


def test_normalize_range_is_zero_to_one():
    result = _normalize([0.0, 0.5, 1.0])
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(0.5)
    assert result[2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# render_sparkline
# ---------------------------------------------------------------------------

def test_render_sparkline_empty_returns_empty_string():
    assert render_sparkline([]) == ""


def test_render_sparkline_single_value_returns_one_char():
    result = render_sparkline([1.0])
    assert len(result) == 1


def test_render_sparkline_ascending_ends_higher_than_start():
    result = render_sparkline([0.0, 0.25, 0.5, 0.75, 1.0])
    assert len(result) == 5
    # Last character should be a 'higher' block than the first
    assert result[-1] >= result[0]


def test_render_sparkline_constant_sequence_uses_lowest_block():
    result = render_sparkline([5.0, 5.0, 5.0])
    # All values normalize to 0 → lowest block character
    assert all(c == result[0] for c in result)


# ---------------------------------------------------------------------------
# build_sparklines
# ---------------------------------------------------------------------------

def test_build_sparklines_empty_trends_returns_empty():
    assert build_sparklines({}) == []


def test_build_sparklines_skips_empty_value_lists():
    rows = build_sparklines({"a": [], "b": [1.0, 2.0]})
    assert len(rows) == 1
    assert rows[0].key == "b"


def test_build_sparklines_sorted_by_range_descending():
    trends = {
        "small": [1.0, 1.1],
        "large": [0.0, 10.0],
        "mid": [2.0, 5.0],
    }
    rows = build_sparklines(trends)
    ranges = [r.max_val - r.min_val for r in rows]
    assert ranges == sorted(ranges, reverse=True)


def test_build_sparklines_top_n_limits_results():
    trends = {f"key{i}": [float(i), float(i + 5)] for i in range(10)}
    rows = build_sparklines(trends, top_n=3)
    assert len(rows) == 3


def test_build_sparklines_row_fields_populated():
    rows = build_sparklines({"w": [1.0, 3.0, 2.0]})
    assert len(rows) == 1
    r = rows[0]
    assert r.key == "w"
    assert r.min_val == pytest.approx(1.0)
    assert r.max_val == pytest.approx(3.0)
    assert len(r.sparkline) == 3


# ---------------------------------------------------------------------------
# format_sparklines
# ---------------------------------------------------------------------------

def test_format_sparklines_empty_returns_placeholder():
    assert format_sparklines([]) == "(no sparkline data)"


def test_format_sparklines_contains_key():
    rows = build_sparklines({"layer.weight": [0.1, 0.5, 0.9]})
    output = format_sparklines(rows)
    assert "layer.weight" in output


def test_format_sparklines_contains_min_max():
    rows = build_sparklines({"x": [2.0, 8.0]})
    output = format_sparklines(rows)
    assert "2.0000" in output
    assert "8.0000" in output
