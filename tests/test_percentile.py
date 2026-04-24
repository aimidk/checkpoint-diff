"""Tests for checkpoint_diff.percentile and checkpoint_diff.cli_percentile."""
from __future__ import annotations

import argparse
import math

import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.percentile import (
    PERCENTILES,
    PercentileRow,
    compute_percentiles,
    format_percentiles,
)
from checkpoint_diff.cli_percentile import add_percentile_args, apply_percentile


def _td(a, b, status="changed"):
    return TensorDiff(
        tensor_a=np.array(a, dtype=float) if a is not None else None,
        tensor_b=np.array(b, dtype=float) if b is not None else None,
        status=status,
    )


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(kwargs)


# ---------------------------------------------------------------------------
# compute_percentiles
# ---------------------------------------------------------------------------

def test_compute_percentiles_returns_row_for_changed_key():
    diff = _make_diff(w=_td(list(range(100)), list(range(1, 101))))
    rows = compute_percentiles(diff)
    assert len(rows) == 1
    assert rows[0].key == "w"


def test_compute_percentiles_skips_removed_key():
    diff = _make_diff(w=_td(list(range(10)), None, status="removed"))
    rows = compute_percentiles(diff)
    assert rows == []


def test_compute_percentiles_added_key_has_nan_in_a():
    diff = _make_diff(w=_td(None, list(range(10)), status="added"))
    rows = compute_percentiles(diff)
    assert len(rows) == 1
    assert all(math.isnan(rows[0].percentiles_a[p]) for p in PERCENTILES)


def test_compute_percentiles_deltas_correct():
    a = [float(i) for i in range(100)]
    b = [v + 1.0 for v in a]
    diff = _make_diff(w=_td(a, b))
    rows = compute_percentiles(diff)
    for p in PERCENTILES:
        assert abs(rows[0].deltas[p] - 1.0) < 1e-6


def test_compute_percentiles_key_filter():
    diff = _make_diff(
        w=_td(list(range(10)), list(range(10))),
        v=_td(list(range(10)), list(range(10))),
    )
    rows = compute_percentiles(diff, keys=["w"])
    assert len(rows) == 1
    assert rows[0].key == "w"


# ---------------------------------------------------------------------------
# format_percentiles
# ---------------------------------------------------------------------------

def test_format_percentiles_empty_returns_message():
    result = format_percentiles([])
    assert "No percentile" in result


def test_format_percentiles_contains_key():
    diff = _make_diff(layer=_td(list(range(50)), list(range(1, 51))))
    rows = compute_percentiles(diff)
    out = format_percentiles(rows)
    assert "layer" in out


def test_format_percentiles_abs_mode_shows_b_label():
    diff = _make_diff(w=_td(list(range(20)), list(range(20))))
    rows = compute_percentiles(diff)
    out = format_percentiles(rows, show_delta=False)
    assert "[b]" in out


def test_format_percentiles_delta_mode_shows_delta_label():
    diff = _make_diff(w=_td(list(range(20)), list(range(20))))
    rows = compute_percentiles(diff)
    out = format_percentiles(rows, show_delta=True)
    assert "[Δ]" in out


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _make_parser():
    p = argparse.ArgumentParser()
    add_percentile_args(p)
    return p


def test_add_percentile_args_registers_flag():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.percentiles is False


def test_add_percentile_args_keys_default_none():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.percentile_keys is None


def test_apply_percentile_returns_none_when_flag_off():
    p = _make_parser()
    ns = p.parse_args([])
    diff = _make_diff(w=_td(list(range(10)), list(range(10))))
    assert apply_percentile(ns, diff) is None


def test_apply_percentile_returns_string_when_flag_on():
    p = _make_parser()
    ns = p.parse_args(["--percentiles"])
    diff = _make_diff(w=_td(list(range(20)), list(range(1, 21))))
    result = apply_percentile(ns, diff)
    assert isinstance(result, str)
    assert "w" in result


def test_apply_percentile_appends_to_output_lines():
    p = _make_parser()
    ns = p.parse_args(["--percentiles"])
    diff = _make_diff(w=_td(list(range(20)), list(range(1, 21))))
    lines = []
    apply_percentile(ns, diff, output_lines=lines)
    assert len(lines) == 1
