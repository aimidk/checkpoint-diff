"""Summary statistics for a CheckpointDiff."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from checkpoint_diff.diff import CheckpointDiff


@dataclass
class DiffSummary:
    total_keys: int
    added: int
    removed: int
    changed: int
    unchanged: int
    max_abs_mean_change: float | None
    most_changed_key: str | None

    def as_dict(self) -> dict:
        return {
            "total_keys": self.total_keys,
            "added": self.added,
            "removed": self.removed,
            "changed": self.changed,
            "unchanged": self.unchanged,
            "max_abs_mean_change": self.max_abs_mean_change,
            "most_changed_key": self.most_changed_key,
        }


def summarize(diff: "CheckpointDiff") -> DiffSummary:
    """Compute a summary of a CheckpointDiff."""
    added = removed = changed = unchanged = 0
    max_change: float | None = None
    most_changed: str | None = None

    for key, td in diff.items():
        if td.status == "added":
            added += 1
        elif td.status == "removed":
            removed += 1
        elif td.status == "changed":
            changed += 1
            if td.mean_delta is not None:
                abs_delta = abs(td.mean_delta)
                if max_change is None or abs_delta > max_change:
                    max_change = abs_delta
                    most_changed = key
        else:
            unchanged += 1

    total = added + removed + changed + unchanged
    return DiffSummary(
        total_keys=total,
        added=added,
        removed=removed,
        changed=changed,
        unchanged=unchanged,
        max_abs_mean_change=max_change,
        most_changed_key=most_changed,
    )


def format_summary(summary: DiffSummary) -> str:
    """Return a human-readable summary string."""
    lines = [
        f"Total keys : {summary.total_keys}",
        f"  Added    : {summary.added}",
        f"  Removed  : {summary.removed}",
        f"  Changed  : {summary.changed}",
        f"  Unchanged: {summary.unchanged}",
    ]
    if summary.most_changed_key is not None:
        lines.append(
            f"Most changed key: {summary.most_changed_key} "
            f"(|Δmean|={summary.max_abs_mean_change:.6g})"
        )
    return "\n".join(lines)
