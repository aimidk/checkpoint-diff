"""Gradient norm analysis for checkpoint diffs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class GradientRow:
    key: str
    l2_norm_a: float
    l2_norm_b: float
    norm_delta: float
    rel_change: float  # (b - a) / (a + eps)


def _l2_norm(arr: Optional[np.ndarray]) -> float:
    if arr is None:
        return float("nan")
    return float(np.sqrt(np.sum(arr ** 2)))


def _rel_change(a: float, b: float, eps: float = 1e-12) -> float:
    if np.isnan(a) or np.isnan(b):
        return float("nan")
    return (b - a) / (abs(a) + eps)


def compute_gradient_norms(
    diff: CheckpointDiff,
    top_n: Optional[int] = None,
) -> List[GradientRow]:
    """Compute L2 gradient norms for each tensor in the diff."""
    rows: List[GradientRow] = []
    for key, td in diff.items():
        if not isinstance(td, TensorDiff):
            continue
        norm_a = _l2_norm(td.array_a)
        norm_b = _l2_norm(td.array_b)
        delta = float("nan") if (np.isnan(norm_a) or np.isnan(norm_b)) else norm_b - norm_a
        rows.append(
            GradientRow(
                key=key,
                l2_norm_a=norm_a,
                l2_norm_b=norm_b,
                norm_delta=delta,
                rel_change=_rel_change(norm_a, norm_b),
            )
        )
    rows.sort(key=lambda r: abs(r.norm_delta) if not np.isnan(r.norm_delta) else 0.0, reverse=True)
    if top_n is not None:
        rows = rows[:top_n]
    return rows


def format_gradient_norms(rows: List[GradientRow]) -> str:
    """Return a human-readable table of gradient norm rows."""
    if not rows:
        return "No gradient norm data."
    header = f"{'Key':<40} {'L2(A)':>12} {'L2(B)':>12} {'Delta':>12} {'Rel%':>8}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        rel_pct = f"{r.rel_change * 100:+.2f}" if not np.isnan(r.rel_change) else "  nan"
        norm_a = f"{r.l2_norm_a:.6f}" if not np.isnan(r.l2_norm_a) else "     nan"
        norm_b = f"{r.l2_norm_b:.6f}" if not np.isnan(r.l2_norm_b) else "     nan"
        delta = f"{r.norm_delta:+.6f}" if not np.isnan(r.norm_delta) else "     nan"
        lines.append(f"{r.key:<40} {norm_a:>12} {norm_b:>12} {delta:>12} {rel_pct:>8}")
    return "\n".join(lines)
