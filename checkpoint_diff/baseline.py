"""Baseline comparison: pin a checkpoint as a reference and compare future checkpoints against it."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

DEFAULT_BASELINE_FILE = ".checkpoint_baseline.json"


def _baseline_path(store_path: Optional[str] = None) -> Path:
    return Path(store_path or DEFAULT_BASELINE_FILE)


def set_baseline(checkpoint_path: str, store_path: Optional[str] = None) -> None:
    """Pin *checkpoint_path* as the current baseline."""
    resolved = str(Path(checkpoint_path).resolve())
    data = {"baseline": resolved}
    _baseline_path(store_path).write_text(json.dumps(data, indent=2))


def get_baseline(store_path: Optional[str] = None) -> Optional[str]:
    """Return the pinned baseline path, or None if none is set."""
    p = _baseline_path(store_path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return data.get("baseline")
    except (json.JSONDecodeError, OSError):
        return None


def clear_baseline(store_path: Optional[str] = None) -> bool:
    """Remove the stored baseline. Returns True if a file was removed."""
    p = _baseline_path(store_path)
    if p.exists():
        p.unlink()
        return True
    return False


def baseline_exists(store_path: Optional[str] = None) -> bool:
    """Return True if a baseline is currently set."""
    return get_baseline(store_path) is not None


def format_baseline_status(store_path: Optional[str] = None) -> str:
    """Return a human-readable string describing the current baseline state."""
    path = get_baseline(store_path)
    if path is None:
        return "No baseline set."
    exists_on_disk = os.path.exists(path)
    status = "(file found)" if exists_on_disk else "(file NOT found on disk)"
    return f"Baseline: {path}  {status}"
