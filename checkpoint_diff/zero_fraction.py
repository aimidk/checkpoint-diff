"""Compute the fraction of exactly-zero elements in each tensor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import math
import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class ZeroFractionRow:
    key: str
    status: str
    zero_frac_a: float   # NaN when tensor absent
    zero_frac_b: float
    delta: float         # zero_frac_b - zero_frac_a; NaN when either is absent


def _zero_frac(arr: Optional[np.ndarray]) -> float:
    if arr is None:
        return math.nan
    flat = arr.ravel()
    if flat.size == 0:
        return math.nan
    return float(np.sum(flat == 0.0)) / flat.size


def compute_zero_fractions(diff: CheckpointDiff) -> List[ZeroFractionRow]:
    """Return a ZeroFractionRow for every key in *diff* (skips unchanged)."""
    rows: List[ZeroFractionRow] = []
    for key, td in diff.items():
        if td.status == "unchanged":
            continue
        za = _zero_frac(td.array_a)
        zb = _zero_frac(td.array_b)
        delta = (zb - za) if not (math.isnan(za) or math.isnan(zb)) else math.nan
        rows.append(ZeroFractionRow(key=key, status=td.status,
                                    zero_frac_a=za, zero_frac_b=zb, delta=delta))
    rows.sort(key=lambda r: abs(r.delta) if not math.isnan(r.delta) else -1,
              reverse=True)
    return rows


def _fmt(v: float) -> str:
    return "n/a" if math.isnan(v) else f"{v:.4f}"


def format_zero_fractions(rows: List[ZeroFractionRow], top_n: Optional[int] = None) -> str:
    if not rows:
        return "No zero-fraction data to display."
    shown = rows[:top_n] if top_n else rows
    header = f"{'Key':<40} {'Status':<10} {'ZeroA':>8} {'ZeroB':>8} {'Delta':>8}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in shown:
        lines.append(
            f"{r.key:<40} {r.status:<10} {_fmt(r.zero_frac_a):>8}"
            f" {_fmt(r.zero_frac_b):>8} {_fmt(r.delta):>8}"
        )
    return "\n".join(lines)
