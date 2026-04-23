"""Compare layer topology (names, shapes, dtypes) between two checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class TopologyRow:
    key: str
    shape_a: Optional[Tuple[int, ...]]
    shape_b: Optional[Tuple[int, ...]]
    dtype_a: Optional[str]
    dtype_b: Optional[str]
    shape_changed: bool
    dtype_changed: bool
    status: str  # 'added' | 'removed' | 'changed' | 'unchanged'


def _dtype_str(arr: Optional[np.ndarray]) -> Optional[str]:
    return str(arr.dtype) if arr is not None else None


def build_topology(diff: CheckpointDiff) -> List[TopologyRow]:
    """Build a topology comparison table from a CheckpointDiff."""
    rows: List[TopologyRow] = []
    for key, td in sorted(diff.items()):
        a, b = td.array_a, td.array_b
        shape_a = tuple(a.shape) if a is not None else None
        shape_b = tuple(b.shape) if b is not None else None
        dtype_a = _dtype_str(a)
        dtype_b = _dtype_str(b)
        shape_changed = shape_a != shape_b
        dtype_changed = dtype_a != dtype_b
        rows.append(
            TopologyRow(
                key=key,
                shape_a=shape_a,
                shape_b=shape_b,
                dtype_a=dtype_a,
                dtype_b=dtype_b,
                shape_changed=shape_changed,
                dtype_changed=dtype_changed,
                status=td.status,
            )
        )
    return rows


def format_topology(rows: List[TopologyRow], show_unchanged: bool = False) -> str:
    """Render topology rows as a human-readable table."""
    if not rows:
        return "(no tensors)"

    visible = [
        r for r in rows
        if show_unchanged or r.status != "unchanged" or r.shape_changed or r.dtype_changed
    ]
    if not visible:
        return "All tensor topologies are identical."

    header = f"{'KEY':<40} {'SHAPE_A':<20} {'SHAPE_B':<20} {'DTYPE_A':<10} {'DTYPE_B':<10} STATUS"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in visible:
        sa = str(r.shape_a) if r.shape_a is not None else "—"
        sb = str(r.shape_b) if r.shape_b is not None else "—"
        da = r.dtype_a or "—"
        db = r.dtype_b or "—"
        flag = "*" if (r.shape_changed or r.dtype_changed) else " "
        lines.append(
            f"{flag}{r.key:<39} {sa:<20} {sb:<20} {da:<10} {db:<10} {r.status}"
        )
    return "\n".join(lines)
