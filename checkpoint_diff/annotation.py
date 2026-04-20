"""Annotation support: attach human-readable notes to tensor diffs."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from checkpoint_diff.diff import CheckpointDiff


@dataclass
class AnnotationStore:
    """Maps tensor keys to free-text annotations."""
    notes: Dict[str, str] = field(default_factory=dict)

    def add(self, key: str, note: str) -> None:
        """Add or overwrite an annotation for *key*."""
        self.notes[key] = note

    def get(self, key: str) -> Optional[str]:
        """Return the annotation for *key*, or None."""
        return self.notes.get(key)

    def remove(self, key: str) -> bool:
        """Delete annotation; return True if it existed."""
        if key in self.notes:
            del self.notes[key]
            return True
        return False


def load_annotations(path: str | Path) -> AnnotationStore:
    """Load an AnnotationStore from a JSON file."""
    data = json.loads(Path(path).read_text())
    return AnnotationStore(notes=data.get("notes", {}))


def save_annotations(store: AnnotationStore, path: str | Path) -> None:
    """Persist an AnnotationStore to a JSON file."""
    Path(path).write_text(json.dumps({"notes": store.notes}, indent=2))


def annotate_report(diff: CheckpointDiff, store: AnnotationStore) -> str:
    """Return a text block listing each diff key with its annotation."""
    lines: list[str] = []
    for key in sorted(diff.keys()):
        note = store.get(key)
        tag = f"  # {note}" if note else ""
        td = diff[key]
        lines.append(f"{key} [{td.status}]{tag}")
    return "\n".join(lines)
