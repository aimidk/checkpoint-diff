"""Watch a directory for new checkpoints and report diffs automatically."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from checkpoint_diff.loader import load_checkpoint
from checkpoint_diff.diff import compute_diff, CheckpointDiff

SUPPORTED_EXTENSIONS = {".pt", ".pth", ".npz", ".npy"}


@dataclass
class WatchState:
    """Tracks the last seen checkpoint in a watched directory."""
    directory: Path
    last_path: Optional[Path] = None
    last_checkpoint: Optional[dict] = None
    diffs_seen: int = 0
    events: list[str] = field(default_factory=list)


def _find_latest_checkpoint(directory: Path) -> Optional[Path]:
    """Return the most recently modified checkpoint file in *directory*."""
    candidates = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix in SUPPORTED_EXTENSIONS
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def poll_once(
    state: WatchState,
    on_diff: Callable[[Path, Path, CheckpointDiff], None],
) -> WatchState:
    """Check for a new checkpoint and call *on_diff* if one is found.

    Returns the (potentially updated) state.
    """
    latest = _find_latest_checkpoint(state.directory)
    if latest is None or latest == state.last_path:
        return state

    current_ckpt = load_checkpoint(str(latest))
    if state.last_checkpoint is not None and state.last_path is not None:
        diff = compute_diff(state.last_checkpoint, current_ckpt)
        on_diff(state.last_path, latest, diff)
        state.diffs_seen += 1
        state.events.append(f"{state.last_path.name} -> {latest.name}")

    state.last_path = latest
    state.last_checkpoint = current_ckpt
    return state


def watch(
    directory: str | Path,
    on_diff: Callable[[Path, Path, CheckpointDiff], None],
    interval: float = 5.0,
    max_polls: Optional[int] = None,
) -> WatchState:
    """Poll *directory* every *interval* seconds, calling *on_diff* on changes.

    Stops after *max_polls* iterations (useful for testing / CI).
    """
    state = WatchState(directory=Path(directory))
    polls = 0
    while True:
        state = poll_once(state, on_diff)
        polls += 1
        if max_polls is not None and polls >= max_polls:
            break
        time.sleep(interval)
    return state
