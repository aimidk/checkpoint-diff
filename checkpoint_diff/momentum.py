"""Momentum analysis: compare effective update magnitudes between checkpoints.

For each key present in both checkpoints, compute the L2 norm of the weight
delta (b - a) and express it relative to the L2 norm of the original weights.
This gives a rough proxy for the effective learning step / momentum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff


@dataclass
class MomentumRow:
    key: str
    delta_l2: float          # ||b - a||_2
    base_l2: float           # ||a||_2
    rel_momentum: float      # delta_l2 / base_l2  (nan when base_l2 == 0)
    status: str


def _l2(arr: Optional[np.ndarray]) -> float:
    if arr is None or arr.size == 0:
        return math.nan
    return float(np.sqrt(np.sum(arr.astype(float) ** 2)))


def _rel(delta: float, base: float) -> float:
    if math.isnan(delta) or math.isnan(base):
        return math.nan
    if base == 0.0:
        return math.nan
    return delta / base


def compute_momentum(diff: CheckpointDiff) -> List[MomentumRow]:
    """Return a MomentumRow for every key that has tensor data in at least one side."""
    rows: List[MomentumRow] = []
    for key, td in diff.items():
        if td.status == "removed":
            continue
        a = td.array_a
        b = td.array_b
        if b is None:
            continue
        if a is not None and a.shape == b.shape:
            delta_arr = b.astype(float) - a.astype(float)
        else:
            delta_arr = b.astype(float)
        d_l2 = _l2(delta_arr)
        b_l2 = _l2(a)
        rows.append(MomentumRow(
            key=key,
            delta_l2=d_l2,
            base_l2=b_l2,
            rel_momentum=_rel(d_l2, b_l2),
            status=td.status,
        ))
    rows.sort(key=lambda r: (math.isnan(r.rel_momentum), -abs(r.rel_momentum) if not math.isnan(r.rel_momentum) else 0))
    return rows


def _fmt(v: float, precision: int = 6) -> str:
    return "nan" if math.isnan(v) else f"{v:.{precision}f}"


def format_momentum(rows: List[MomentumRow], top_n: Optional[int] = None) -> str:
    if not rows:
        return "No momentum data available."
    display = rows[:top_n] if top_n else rows
    header = f"{'Key':<40} {'delta_L2':>12} {'base_L2':>12} {'rel_momentum':>14} {'status':>10}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in display:
        lines.append(
            f"{r.key:<40} {_fmt(r.delta_l2):>12} {_fmt(r.base_l2):>12} {_fmt(r.rel_momentum):>14} {r.status:>10}"
        )
    return "\n".join(lines)
