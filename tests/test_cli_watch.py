"""Tests for checkpoint_diff.cli_watch."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from checkpoint_diff.cli_watch import add_watch_args, apply_watch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_watch_args(p)
    p.add_argument("--verbose", action="store_true", default=False)
    return p


def _write_npz(path: Path, value: float) -> None:
    np.savez(str(path), weight=np.array([value], dtype=np.float32))


# ---------------------------------------------------------------------------
# add_watch_args
# ---------------------------------------------------------------------------

def test_add_watch_args_registers_watch() -> None:
    parser = _make_parser()
    args = parser.parse_args([])
    assert hasattr(args, "watch")
    assert args.watch is None


def test_add_watch_args_interval_default() -> None:
    parser = _make_parser()
    args = parser.parse_args([])
    assert args.watch_interval == 5.0


def test_add_watch_args_max_polls_default() -> None:
    parser = _make_parser()
    args = parser.parse_args([])
    assert args.watch_max_polls is None


def test_add_watch_args_parses_watch_dir(tmp_path: Path) -> None:
    parser = _make_parser()
    args = parser.parse_args(["--watch", str(tmp_path)])
    assert args.watch == str(tmp_path)


def test_add_watch_args_parses_custom_interval() -> None:
    """Ensure --watch-interval accepts float values other than the default."""
    parser = _make_parser()
    args = parser.parse_args(["--watch-interval", "2.5"])
    assert args.watch_interval == 2.5


def test_add_watch_args_parses_max_polls() -> None:
    """Ensure --watch-max-polls is stored as an integer."""
    parser = _make_parser()
    args = parser.parse_args(["--watch-max-polls", "10"])
    assert args.watch_max_polls == 10


# ---------------------------------------------------------------------------
# apply_watch
# ---------------------------------------------------------------------------

def test_apply_watch_returns_false_when_not_set() -> None:
    parser = _make_parser()
    args = parser.parse_args([])
    assert apply_watch(args) is False


def test_apply_watch_returns_true_and_calls_watch(tmp_path: Path) -> None:
    _write_npz(tmp_path / "ckpt.npz", 1.0)
    parser = _make_parser()
    args = parser.parse_args([
        "--watch", str(tmp_path),
        "--watch-interval", "0",
        "--watch-max-polls", "1",
    ])
    result = apply_watch(args)
    assert result is True


def test_apply_watch_verbose_flag_passed(tmp_path: Path) -> None:
    _write_npz(tmp_path / "ckpt.npz", 0.0)
    parser = _make_parser()
    args = parser.parse_args([
        "--watch", str(tmp_path),
        "--watch-interval", "0",
        "--watch-max-polls", "1",
        "--verbose",
    ])
    with patch("checkpoint_diff.cli_watch.watch") as mock_watch:
        mock_watch.return_value = MagicMock()
        apply_watch(args)
        _, kwargs = mock_watch.call_args
        assert kwargs["interval"] == 0.0
        assert kwargs["max_polls"] == 1
