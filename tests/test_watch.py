"""Tests for checkpoint_diff.watch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from checkpoint_diff.watch import WatchState, poll_once, watch, _find_latest_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_npz(path: Path, value: float) -> None:
    np.savez(str(path), weight=np.array([value], dtype=np.float32))


# ---------------------------------------------------------------------------
# _find_latest_checkpoint
# ---------------------------------------------------------------------------

def test_find_latest_returns_none_for_empty_dir(tmp_path: Path) -> None:
    assert _find_latest_checkpoint(tmp_path) is None


def test_find_latest_ignores_unsupported_extensions(tmp_path: Path) -> None:
    (tmp_path / "model.txt").write_text("nope")
    assert _find_latest_checkpoint(tmp_path) is None


def test_find_latest_returns_most_recent(tmp_path: Path) -> None:
    p1 = tmp_path / "ckpt_1.npz"
    p2 = tmp_path / "ckpt_2.npz"
    _write_npz(p1, 1.0)
    _write_npz(p2, 2.0)
    # Force p2 to have a newer mtime
    p2.touch()
    assert _find_latest_checkpoint(tmp_path) == p2


# ---------------------------------------------------------------------------
# poll_once
# ---------------------------------------------------------------------------

def test_poll_once_no_checkpoint_returns_unchanged_state(tmp_path: Path) -> None:
    state = WatchState(directory=tmp_path)
    new_state = poll_once(state, lambda *a: None)
    assert new_state.last_path is None
    assert new_state.diffs_seen == 0


def test_poll_once_first_checkpoint_sets_last_path(tmp_path: Path) -> None:
    p = tmp_path / "ckpt.npz"
    _write_npz(p, 1.0)
    state = WatchState(directory=tmp_path)
    state = poll_once(state, lambda *a: None)
    assert state.last_path == p
    assert state.diffs_seen == 0  # no previous checkpoint to diff against


def test_poll_once_second_checkpoint_triggers_callback(tmp_path: Path) -> None:
    p1 = tmp_path / "ckpt_1.npz"
    _write_npz(p1, 1.0)

    state = WatchState(directory=tmp_path)
    state = poll_once(state, lambda *a: None)  # seed with p1

    p2 = tmp_path / "ckpt_2.npz"
    _write_npz(p2, 2.0)
    p2.touch()  # ensure newer mtime

    calls: list = []
    state = poll_once(state, lambda prev, curr, diff: calls.append((prev, curr)))

    assert len(calls) == 1
    assert calls[0][0] == p1
    assert calls[0][1] == p2
    assert state.diffs_seen == 1


# ---------------------------------------------------------------------------
# watch (max_polls)
# ---------------------------------------------------------------------------

def test_watch_runs_max_polls_times(tmp_path: Path) -> None:
    _write_npz(tmp_path / "ckpt.npz", 0.0)
    state = watch(tmp_path, on_diff=lambda *a: None, interval=0.0, max_polls=3)
    # Should not raise; state is returned
    assert isinstance(state, WatchState)


def test_watch_records_events(tmp_path: Path) -> None:
    p1 = tmp_path / "ckpt_1.npz"
    p2 = tmp_path / "ckpt_2.npz"
    _write_npz(p1, 1.0)
    _write_npz(p2, 2.0)
    p2.touch()

    # Two polls: first seeds p2 (latest), second finds no new file
    state = watch(tmp_path, on_diff=lambda *a: None, interval=0.0, max_polls=2)
    # No diff triggered because both files existed before first poll
    assert isinstance(state.events, list)
