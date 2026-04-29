"""Per-layer weight norm tracking: L∞ and Frobenius norms with delta."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import math
import numpy as np

from checkpoint_diff.diff import CheckpointDiff


@dataclass
class WeightNormRow:
    key: str
    linf_a: float
    linf_b: float
    frob_a: float
    frob_b: float
    linf_delta: float
    frob_delta: float


def _linf(arr: Optional[np.ndarray]) -> float:
    if arr is None or arr.size == 0:
        return math.nan
    return float(np.max(np.abs(arr)))


def _frob(arr: Optional[np.ndarray]) -> float:
    if arr is None or arr.size == 0:
        return math.nan
    return float(np.sqrt(np.sum(arr.astype(float) ** 2)))


def _delta(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b):
        return math.nan
    return b - a


def compute_weight_norms(diff: CheckpointDiff) -> List[WeightNormRow]:
    rows: List[WeightNormRow] = []
    for key, td in diff.tensors.items():
        if td.status == "removed":
            continue
        la = _linf(td.array_a)
        lb = _linf(td.array_b)
        fa = _frob(td.array_a)
        fb = _frob(td.array_b)
        rows.append(
            WeightNormRow(
                key=key,
                linf_a=la,
                linf_b=lb,
                frob_a=fa,
                frob_b=fb,
                linf_delta=_delta(la, lb),
                frob_delta=_delta(fa, fb),
            )
        )
    rows.sort(key=lambda r: abs(r.frob_delta) if not math.isnan(r.frob_delta) else 0.0, reverse=True)
    return rows


def _fmt(v: float) -> str:
    return "n/a" if math.isnan(v) else f"{v:.6g}"


def format_weight_norms(rows: List[WeightNormRow], top_n: Optional[int] = None) -> str:
    if not rows:
        return "No weight norm data available."
    display = rows[:top_n] if top_n else rows
    header = f"{'Key':<40} {'L∞(a)':>10} {'L∞(b)':>10} {'ΔL∞':>10} {'Frob(a)':>10} {'Frob(b)':>10} {'ΔFrob':>10}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in display:
        lines.append(
            f"{r.key:<40} {_fmt(r.linf_a):>10} {_fmt(r.linf_b):>10} {_fmt(r.linf_delta):>10}"
            f" {_fmt(r.frob_a):>10} {_fmt(r.frob_b):>10} {_fmt(r.frob_delta):>10}"
        )
    return "\n".join(lines)
