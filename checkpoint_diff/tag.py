"""Tag tensors in a checkpoint diff with user-defined labels."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from checkpoint_diff.diff import CheckpointDiff


@dataclass
class TagStore:
    """Maps tensor keys to lists of string tags."""
    _data: Dict[str, List[str]] = field(default_factory=dict)

    def add(self, key: str, tag: str) -> None:
        """Add *tag* to *key*, ignoring duplicates."""
        tags = self._data.setdefault(key, [])
        if tag not in tags:
            tags.append(tag)

    def get(self, key: str) -> List[str]:
        """Return tags for *key* (empty list if none)."""
        return list(self._data.get(key, []))

    def remove(self, key: str, tag: str) -> bool:
        """Remove *tag* from *key*. Returns True if it existed."""
        tags = self._data.get(key, [])
        if tag in tags:
            tags.remove(tag)
            return True
        return False

    def all_tags(self) -> List[str]:
        """Return sorted list of all unique tags across all keys."""
        return sorted({t for tags in self._data.values() for t in tags})

    def keys_with_tag(self, tag: str) -> List[str]:
        """Return all keys that have *tag*."""
        return [k for k, tags in self._data.items() if tag in tags]


def filter_diff_by_tag(diff: CheckpointDiff, store: TagStore, tag: str) -> CheckpointDiff:
    """Return a new CheckpointDiff keeping only entries whose key has *tag*."""
    tagged_keys = set(store.keys_with_tag(tag))
    return CheckpointDiff(
        added={k: v for k, v in diff.added.items() if k in tagged_keys},
        removed={k: v for k, v in diff.removed.items() if k in tagged_keys},
        changed={k: v for k, v in diff.changed.items() if k in tagged_keys},
        unchanged={k: v for k, v in diff.unchanged.items() if k in tagged_keys},
    )


def format_tags(store: TagStore, keys: Optional[List[str]] = None) -> str:
    """Return a human-readable table of key -> tags."""
    keys = keys if keys is not None else sorted(store._data.keys())
    if not keys:
        return "(no tags)"
    lines = []
    for k in keys:
        tags = store.get(k)
        tag_str = ", ".join(tags) if tags else "(none)"
        lines.append(f"  {k}: [{tag_str}]")
    return "\n".join(lines)
