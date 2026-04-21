"""Tests for checkpoint_diff.baseline and checkpoint_diff.cli_baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from checkpoint_diff.baseline import (
    baseline_exists,
    clear_baseline,
    format_baseline_status,
    get_baseline,
    set_baseline,
)
from checkpoint_diff.cli_baseline import add_baseline_args, apply_baseline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store(tmp_path: Path) -> str:
    return str(tmp_path / "baseline.json")


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_baseline_args(p)
    return p


# ---------------------------------------------------------------------------
# baseline.py
# ---------------------------------------------------------------------------


def test_set_and_get_baseline(tmp_path):
    store = _store(tmp_path)
    ckpt = str(tmp_path / "model.npz")
    set_baseline(ckpt, store)
    result = get_baseline(store)
    # get_baseline returns the resolved absolute path
    assert result is not None
    assert result.endswith("model.npz")


def test_get_baseline_returns_none_when_unset(tmp_path):
    assert get_baseline(_store(tmp_path)) is None


def test_clear_baseline_returns_true_when_file_existed(tmp_path):
    store = _store(tmp_path)
    set_baseline("/some/path.npz", store)
    assert clear_baseline(store) is True
    assert get_baseline(store) is None


def test_clear_baseline_returns_false_when_no_file(tmp_path):
    assert clear_baseline(_store(tmp_path)) is False


def test_baseline_exists_true_after_set(tmp_path):
    store = _store(tmp_path)
    set_baseline("/some/path.npz", store)
    assert baseline_exists(store) is True


def test_baseline_exists_false_when_unset(tmp_path):
    assert baseline_exists(_store(tmp_path)) is False


def test_format_baseline_status_no_baseline(tmp_path):
    msg = format_baseline_status(_store(tmp_path))
    assert "No baseline" in msg


def test_format_baseline_status_with_missing_file(tmp_path):
    store = _store(tmp_path)
    set_baseline("/nonexistent/model.npz", store)
    msg = format_baseline_status(store)
    assert "NOT found" in msg


# ---------------------------------------------------------------------------
# cli_baseline.py
# ---------------------------------------------------------------------------


def test_add_baseline_args_registers_set_baseline():
    p = _make_parser()
    args = p.parse_args(["--set-baseline", "/tmp/a.npz"])
    assert args.set_baseline == "/tmp/a.npz"


def test_add_baseline_args_clear_flag_default_false():
    p = _make_parser()
    args = p.parse_args([])
    assert args.clear_baseline is False


def test_add_baseline_args_show_flag_default_false():
    p = _make_parser()
    args = p.parse_args([])
    assert args.show_baseline is False


def test_apply_baseline_falls_back_to_stored(tmp_path):
    store = _store(tmp_path)
    set_baseline("/stored/model.npz", store)
    p = _make_parser()
    args = p.parse_args(["--baseline-store", store])
    result = apply_baseline(args, checkpoint_b=None)
    assert result is not None
    assert result.endswith("model.npz")


def test_apply_baseline_explicit_checkpoint_b_takes_priority(tmp_path):
    store = _store(tmp_path)
    set_baseline("/stored/model.npz", store)
    p = _make_parser()
    args = p.parse_args(["--baseline-store", store])
    result = apply_baseline(args, checkpoint_b="/explicit/other.npz")
    assert result == "/explicit/other.npz"
