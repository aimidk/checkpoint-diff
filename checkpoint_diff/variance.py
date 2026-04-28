"""Per-key variance and coefficient-of-variation analysis for checkpoint diffs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import math
import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class VarianceRow:
    key: str
    var_a: float
    var_b: float
    cv_a: float   # coefficient of variation: std / |mean|
    cv_b: float
    var_delta: float  # var_b - var_a


def _variance(arr: Optional[np.ndarray]) -> float:
    if arr is None or arr.size == 0:
        return math.nan
    return float(np.var(arr.astype(float)))


def _cv(arr: Optional[np.ndarray]) -> float:
    """Coefficient of variation: std / |mean|.  Returns nan when mean ~ 0."""
    if arr is None or arr.size == 0:
        return math.nan
    a = arr.astype(float)
    mean = float(np.mean(a))
    if math.isclose(mean, 0.0, abs_tol=1e-12):
        return math.nan
    return float(np.std(a) / abs(mean))


def compute_variance(diff: CheckpointDiff, include_unchanged: bool = False) -> List[VarianceRow]:
    """Return a VarianceRow for every relevant key in *diff*."""
    rows: List[VarianceRow] = []
    for key, td in diff.items():
        if td.status == "removed":
            continue
        if td.status == "unchanged" and not include_unchanged:
            continue
        va = _variance(td.array_a)
        vb = _variance(td.array_b)
        rows.append(
            VarianceRow(
                key=key,
                var_a=va,
                var_b=vb,
                cv_a=_cv(td.array_a),
                cv_b=_cv(td.array_b),
                var_delta=vb - va if not (math.isnan(va) or math.isnan(vb)) else math.nan,
            )
        )
    rows.sort(key=lambda r: abs(r.var_delta) if not math.isnan(r.var_delta) else 0.0, reverse=True)
    return rows


def format_variance(rows: List[VarianceRow], top_n: Optional[int] = None) -> str:
    """Return a human-readable table of variance rows."""
    if not rows:
        return "No variance data available."
    shown = rows[:top_n] if top_n else rows
    header = f"{'Key':<40} {'Var(A)':>12} {'Var(B)':>12} {'ΔVar':>12} {'CV(A)':>9} {'CV(B)':>9}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in shown:
        def _f(v: float) -> str:
            return f"{v:>12.4e}" if not math.isnan(v) else f"{'nan':>12}"
        def _fc(v: float) -> str:
            return f"{v:>9.4f}" if not math.isnan(v) else f"{'nan':>9}"
        lines.append(f"{r.key:<40}{_f(r.var_a)}{_f(r.var_b)}{_f(r.var_delta)}{_fc(r.cv_a)}{_fc(r.cv_b)}")
    return "\n".join(lines)
