"""Group checkpoint diff keys by prefix or custom mapping."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class GroupResult:
    group_name: str
    keys: List[str] = field(default_factory=list)
    diffs: Dict[str, TensorDiff] = field(default_factory=dict)

    @property
    def num_changed(self) -> int:
        return sum(1 for d in self.diffs.values() if d.status == "changed")

    @property
    def num_added(self) -> int:
        return sum(1 for d in self.diffs.values() if d.status == "added")

    @property
    def num_removed(self) -> int:
        return sum(1 for d in self.diffs.values() if d.status == "removed")


def group_by_prefix(diff: CheckpointDiff, sep: str = ".") -> Dict[str, GroupResult]:
    """Group diff entries by the first component of each key (split by *sep*)."""
    groups: Dict[str, GroupResult] = {}
    for key, td in diff.items():
        prefix = key.split(sep)[0] if sep in key else key
        if prefix not in groups:
            groups[prefix] = GroupResult(group_name=prefix)
        groups[prefix].keys.append(key)
        groups[prefix].diffs[key] = td
    return groups


def group_by_map(
    diff: CheckpointDiff, mapping: Dict[str, str], default_group: str = "other"
) -> Dict[str, GroupResult]:
    """Group diff entries using an explicit key->group_name *mapping*."""
    groups: Dict[str, GroupResult] = {}
    for key, td in diff.items():
        group_name = mapping.get(key, default_group)
        if group_name not in groups:
            groups[group_name] = GroupResult(group_name=group_name)
        groups[group_name].keys.append(key)
        groups[group_name].diffs[key] = td
    return groups


def format_groups(groups: Dict[str, GroupResult]) -> str:
    """Return a human-readable summary table of grouped diff results."""
    lines: List[str] = []
    header = f"{'Group':<30} {'Keys':>6} {'Changed':>8} {'Added':>7} {'Removed':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, gr in sorted(groups.items()):
        lines.append(
            f"{name:<30} {len(gr.keys):>6} {gr.num_changed:>8} {gr.num_added:>7} {gr.num_removed:>9}"
        )
    return "\n".join(lines)
