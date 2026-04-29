"""Skewness analysis for tensor distributions across checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff


@dataclass
class SkewnessRow:
    key: str
    skew_a: float
    skew_b: float
    delta: float


def _skewness(arr: Optional[np.ndarray]) -> float:
    """Compute Fisher-Pearson skewness coefficient."""
    if arr is None or arr.size == 0:
        return float("nan")
    flat = arr.astype(float).ravel()
    if flat.size < 3:
        return float("nan")
    mu = flat.mean()
    sigma = flat.std()
    if sigma == 0.0:
        return float("nan")
    return float(np.mean(((flat - mu) / sigma) ** 3))


def compute_skewness(diff: CheckpointDiff) -> list[SkewnessRow]:
    """Return a SkewnessRow for every non-removed tensor in *diff*."""
    rows: list[SkewnessRow] = []
    for key, td in diff.tensors.items():
        if td.status == "removed":
            continue
        skew_a = _skewness(td.a)
        skew_b = _skewness(td.b)
        delta = (
            skew_b - skew_a
            if not (np.isnan(skew_a) or np.isnan(skew_b))
            else float("nan")
        )
        rows.append(SkewnessRow(key=key, skew_a=skew_a, skew_b=skew_b, delta=delta))
    rows.sort(key=lambda r: abs(r.delta) if not np.isnan(r.delta) else 0.0, reverse=True)
    return rows


def format_skewness(rows: list[SkewnessRow], top_n: Optional[int] = None) -> str:
    """Render skewness rows as a plain-text table."""
    if not rows:
        return "No skewness data available."
    if top_n is not None:
        rows = rows[:top_n]
    header = f"{'Key':<40} {'Skew-A':>10} {'Skew-B':>10} {'Delta':>10}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        def _fmt(v: float) -> str:
            return f"{v:>10.4f}" if not np.isnan(v) else f"{'nan':>10}"
        lines.append(f"{r.key:<40}{_fmt(r.skew_a)}{_fmt(r.skew_b)}{_fmt(r.delta)}")
    return "\n".join(lines)
