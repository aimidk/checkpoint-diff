"""Bookmark specific tensor keys for quick re-inspection across runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


class BookmarkStore:
    """Persistent store mapping bookmark labels to tensor keys."""

    def __init__(self) -> None:
        self._data: Dict[str, List[str]] = {}

    def add(self, label: str, key: str) -> None:
        """Add *key* under *label*, ignoring duplicates."""
        self._data.setdefault(label, [])
        if key not in self._data[label]:
            self._data[label].append(key)

    def get(self, label: str) -> List[str]:
        """Return all keys for *label*, or empty list if unknown."""
        return list(self._data.get(label, []))

    def remove(self, label: str, key: str) -> bool:
        """Remove *key* from *label*. Returns True if removed."""
        keys = self._data.get(label, [])
        if key in keys:
            keys.remove(key)
            if not keys:
                del self._data[label]
            return True
        return False

    def labels(self) -> List[str]:
        """Return all known labels."""
        return list(self._data.keys())

    def all_entries(self) -> Dict[str, List[str]]:
        return {k: list(v) for k, v in self._data.items()}


def load_bookmarks(path: str) -> BookmarkStore:
    """Load a BookmarkStore from a JSON file."""
    store = BookmarkStore()
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Bookmark file must contain a JSON object.")
    for label, keys in data.items():
        if not isinstance(keys, list):
            raise ValueError(f"Keys for label '{label}' must be a list.")
        for k in keys:
            store.add(label, k)
    return store


def save_bookmarks(store: BookmarkStore, path: str) -> None:
    """Persist *store* to a JSON file."""
    Path(path).write_text(json.dumps(store.all_entries(), indent=2))


def filter_by_bookmark(
    diff: "CheckpointDiff",  # noqa: F821
    store: BookmarkStore,
    label: str,
) -> Optional[Dict]:
    """Return subset of *diff* entries whose keys are bookmarked under *label*."""
    from checkpoint_diff.diff import CheckpointDiff

    keys = store.get(label)
    if not keys:
        return {}
    return {k: v for k, v in diff.items() if k in keys}
