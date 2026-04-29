"""Tests for checkpoint_diff.zero_fraction and checkpoint_diff.cli_zero_fraction."""
from __future__ import annotations

import argparse
import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff, CheckpointDiff
from checkpoint_diff.zero_fraction import (
    _zero_frac,
    compute_zero_fractions,
    format_zero_fractions,
)
from checkpoint_diff.cli_zero_fraction import add_zero_fraction_args, apply_zero_fraction


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _td(a, b, status="changed") -> TensorDiff:
    return TensorDiff(array_a=a, array_b=b, status=status)


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(kwargs)


# ---------------------------------------------------------------------------
# _zero_frac
# ---------------------------------------------------------------------------

def test_zero_frac_all_zeros():
    assert _zero_frac(np.zeros(10)) == pytest.approx(1.0)


def test_zero_frac_no_zeros():
    assert _zero_frac(np.ones(10)) == pytest.approx(0.0)


def test_zero_frac_mixed():
    arr = np.array([0.0, 1.0, 0.0, 2.0])
    assert _zero_frac(arr) == pytest.approx(0.5)


def test_zero_frac_none_returns_nan():
    assert math.isnan(_zero_frac(None))


def test_zero_frac_empty_returns_nan():
    assert math.isnan(_zero_frac(np.array([])))


# ---------------------------------------------------------------------------
# compute_zero_fractions
# ---------------------------------------------------------------------------

def test_unchanged_keys_excluded():
    diff = _make_diff(w=_td(np.ones(4), np.ones(4), status="unchanged"))
    rows = compute_zero_fractions(diff)
    assert rows == []


def test_changed_key_included():
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.0])
    diff = _make_diff(w=_td(a, b))
    rows = compute_zero_fractions(diff)
    assert len(rows) == 1
    assert rows[0].key == "w"
    assert rows[0].zero_frac_a == pytest.approx(0.5)
    assert rows[0].zero_frac_b == pytest.approx(1.0)
    assert rows[0].delta == pytest.approx(0.5)


def test_added_key_has_nan_in_a():
    diff = _make_diff(w=_td(None, np.ones(4), status="added"))
    rows = compute_zero_fractions(diff)
    assert math.isnan(rows[0].zero_frac_a)
    assert math.isnan(rows[0].delta)


def test_rows_sorted_by_abs_delta_descending():
    diff = _make_diff(
        small=_td(np.array([0.0, 1.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0])),
        large=_td(np.array([1.0, 1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0, 0.0])),
    )
    rows = compute_zero_fractions(diff)
    assert rows[0].key == "large"


# ---------------------------------------------------------------------------
# format_zero_fractions
# ---------------------------------------------------------------------------

def test_format_empty_returns_message():
    assert "No zero-fraction" in format_zero_fractions([])


def test_format_contains_key():
    diff = _make_diff(bias=_td(np.zeros(4), np.ones(4)))
    rows = compute_zero_fractions(diff)
    out = format_zero_fractions(rows)
    assert "bias" in out


def test_format_top_n_limits_rows():
    diff = _make_diff(
        a=_td(np.ones(4), np.zeros(4)),
        b=_td(np.ones(4), np.zeros(4)),
        c=_td(np.ones(4), np.zeros(4)),
    )
    rows = compute_zero_fractions(diff)
    out = format_zero_fractions(rows, top_n=2)
    # header + sep + 2 data rows
    assert len(out.strip().splitlines()) == 4


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_zero_fraction_args(p)
    return p


def test_add_zero_fraction_args_registers_flag():
    p = _make_parser()
    args = p.parse_args(["--zero-fraction"])
    assert args.zero_fraction is True


def test_add_zero_fraction_args_default_false():
    p = _make_parser()
    args = p.parse_args([])
    assert args.zero_fraction is False


def test_add_zero_fraction_top_n_default_none():
    p = _make_parser()
    args = p.parse_args([])
    assert args.zero_fraction_top_n is None


def test_apply_zero_fraction_prints_when_flag_set():
    diff = _make_diff(w=_td(np.ones(4), np.zeros(4)))
    p = _make_parser()
    args = p.parse_args(["--zero-fraction"])
    mock_print = MagicMock()
    apply_zero_fraction(args, diff, print_fn=mock_print)
    mock_print.assert_called_once()


def test_apply_zero_fraction_silent_when_flag_not_set():
    diff = _make_diff(w=_td(np.ones(4), np.zeros(4)))
    p = _make_parser()
    args = p.parse_args([])
    mock_print = MagicMock()
    apply_zero_fraction(args, diff, print_fn=mock_print)
    mock_print.assert_not_called()
