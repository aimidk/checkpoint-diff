"""Per-tensor magnitude analysis: L1/L2 norms and relative change."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .diff import CheckpointDiff, TensorDiff


@dataclass
class MagnitudeRow:
    key: str
    norm_l1_a: float
    norm_l1_b: float
    norm_l2_a: float
    norm_l2_b: float
    rel_change_l2: Optional[float]  # None when a is zero or key is added/removed
    status: str


def _l1(arr: np.ndarray) -> float:
    return float(np.sum(np.abs(arr)))


def _l2(arr: np.ndarray) -> float:
    return float(np.sqrt(np.sum(arr.astype(float) ** 2)))


def _rel_change(a: float, b: float) -> Optional[float]:
    if a == 0.0:
        return None
    return (b - a) / a


def compute_magnitude(diff: CheckpointDiff) -> List[MagnitudeRow]:
    rows: List[MagnitudeRow] = []
    for key, td in diff.items():
        if td.status == "removed":
            rows.append(MagnitudeRow(key, _l1(td.a), 0.0, _l2(td.a), 0.0, None, td.status))
        elif td.status == "added":
            rows.append(MagnitudeRow(key, 0.0, _l1(td.b), 0.0, _l2(td.b), None, td.status))
        else:
            l1a, l1b = _l1(td.a), _l1(td.b)
            l2a, l2b = _l2(td.a), _l2(td.b)
            rows.append(MagnitudeRow(key, l1a, l1b, l2a, l2b, _rel_change(l2a, l2b), td.status))
    rows.sort(key=lambda r: abs(r.rel_change_l2) if r.rel_change_l2 is not None else 0.0, reverse=True)
    return rows


def format_magnitude(rows: List[MagnitudeRow], top_n: Optional[int] = None) -> str:
    if not rows:
        return "No magnitude data available."
    display = rows[:top_n] if top_n else rows
    header = f"{'key':<40} {'status':<10} {'L2(a)':>12} {'L2(b)':>12} {'rel_chg':>10}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in display:
        rel = f"{r.rel_change_l2:+.4f}" if r.rel_change_l2 is not None else "    n/a"
        lines.append(f"{r.key:<40} {r.status:<10} {r.norm_l2_a:>12.4f} {r.norm_l2_b:>12.4f} {rel:>10}")
    return "\n".join(lines)
