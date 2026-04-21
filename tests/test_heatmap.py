"""Tests for checkpoint_diff.heatmap and checkpoint_diff.cli_heatmap."""
from __future__ import annotations

import argparse
import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff, CheckpointDiff
from checkpoint_diff.heatmap import build_heatmap, format_heatmap, _render_bar
from checkpoint_diff.cli_heatmap import add_heatmap_args, apply_heatmap


def _td(mean_a, mean_b, status="changed") -> TensorDiff:
    return TensorDiff(
        status=status,
        shape_a=(4,),
        shape_b=(4,),
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=0.1,
        std_b=0.1,
    )


def _make_diff(**kwargs) -> CheckpointDiff:
    return CheckpointDiff(kwargs)


# --- _render_bar ---

def test_render_bar_full_value():
    bar = _render_bar(1.0, 1.0, width=10)
    assert len(bar) == 10


def test_render_bar_zero_max_returns_spaces():
    bar = _render_bar(0.0, 0.0, width=5)
    assert len(bar) == 5


# --- build_heatmap ---

def test_build_heatmap_sorted_descending():
    diff = _make_diff(
        layer1=_td(0.0, 1.0),
        layer2=_td(0.0, 5.0),
        layer3=_td(0.0, 2.0),
    )
    rows = build_heatmap(diff)
    deltas = [r.abs_mean_delta for r in rows]
    assert deltas == sorted(deltas, reverse=True)


def test_build_heatmap_excludes_unchanged_by_default():
    diff = _make_diff(
        changed=_td(0.0, 1.0, status="changed"),
        same=_td(1.0, 1.0, status="unchanged"),
    )
    rows = build_heatmap(diff)
    keys = [r.key for r in rows]
    assert "same" not in keys
    assert "changed" in keys


def test_build_heatmap_includes_unchanged_when_requested():
    diff = _make_diff(
        changed=_td(0.0, 1.0, status="changed"),
        same=_td(1.0, 1.0, status="unchanged"),
    )
    rows = build_heatmap(diff, include_unchanged=True)
    keys = [r.key for r in rows]
    assert "same" in keys


def test_build_heatmap_top_n_limits_rows():
    diff = _make_diff(
        a=_td(0.0, 3.0),
        b=_td(0.0, 1.0),
        c=_td(0.0, 5.0),
    )
    rows = build_heatmap(diff, top_n=2)
    assert len(rows) == 2


def test_build_heatmap_bars_populated():
    diff = _make_diff(layer=_td(0.0, 2.0))
    rows = build_heatmap(diff)
    assert rows[0].bar != ""


# --- format_heatmap ---

def test_format_heatmap_empty_returns_message():
    assert "no data" in format_heatmap([])


def test_format_heatmap_contains_key_name():
    diff = _make_diff(my_layer=_td(0.0, 1.5))
    rows = build_heatmap(diff)
    output = format_heatmap(rows)
    assert "my_layer" in output


# --- CLI integration ---

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_heatmap_args(p)
    return p


def test_add_heatmap_args_registers_flag():
    p = _make_parser()
    ns = p.parse_args(["--heatmap"])
    assert ns.heatmap is True


def test_add_heatmap_args_top_default_none():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.heatmap_top is None


def test_apply_heatmap_returns_none_when_flag_not_set():
    p = _make_parser()
    ns = p.parse_args([])
    diff = _make_diff(layer=_td(0.0, 1.0))
    assert apply_heatmap(ns, diff) is None


def test_apply_heatmap_returns_string_when_flag_set():
    p = _make_parser()
    ns = p.parse_args(["--heatmap"])
    diff = _make_diff(layer=_td(0.0, 1.0))
    result = apply_heatmap(ns, diff)
    assert isinstance(result, str)
    assert "layer" in result
