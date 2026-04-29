"""Detect dead (always-zero) and saturated neurons across tensor diffs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class DeadNeuronRow:
    key: str
    total_a: int
    dead_a: int
    dead_pct_a: float
    total_b: int
    dead_b: int
    dead_pct_b: float
    dead_delta: float  # dead_pct_b - dead_pct_a


def _dead_count(arr: Optional[np.ndarray], eps: float) -> tuple[int, int]:
    """Return (total_elements, dead_elements) where dead means |v| < eps."""
    if arr is None or arr.size == 0:
        return 0, 0
    flat = arr.flatten().astype(float)
    return int(flat.size), int(np.sum(np.abs(flat) < eps))


def compute_dead_neurons(
    diff: CheckpointDiff,
    eps: float = 1e-6,
    only_changed: bool = False,
) -> List[DeadNeuronRow]:
    """Compute dead-neuron statistics for each tensor in the diff.

    Parameters
    ----------
    diff:
        The checkpoint diff to analyse.
    eps:
        Absolute threshold below which a value is considered zero / dead.
    only_changed:
        When True, only include keys whose dead-neuron percentage changed.
    """
    rows: List[DeadNeuronRow] = []
    for key, td in diff.items():
        if td.status == "removed":
            continue
        total_a, dead_a = _dead_count(td.array_a, eps)
        total_b, dead_b = _dead_count(td.array_b, eps)
        pct_a = dead_a / total_a if total_a else float("nan")
        pct_b = dead_b / total_b if total_b else float("nan")
        delta = pct_b - pct_a if not (np.isnan(pct_a) or np.isnan(pct_b)) else float("nan")
        if only_changed and not np.isnan(delta) and abs(delta) == 0.0:
            continue
        rows.append(
            DeadNeuronRow(
                key=key,
                total_a=total_a,
                dead_a=dead_a,
                dead_pct_a=round(pct_a, 6),
                total_b=total_b,
                dead_b=dead_b,
                dead_pct_b=round(pct_b, 6),
                dead_delta=round(delta, 6),
            )
        )
    rows.sort(key=lambda r: abs(r.dead_delta) if not np.isnan(r.dead_delta) else -1, reverse=True)
    return rows


def format_dead_neurons(rows: List[DeadNeuronRow], top_n: Optional[int] = None) -> str:
    """Render dead-neuron rows as a plain-text table."""
    if not rows:
        return "No dead-neuron data available."
    subset = rows[:top_n] if top_n else rows
    header = f"{'Key':<40} {'dead_a%':>8} {'dead_b%':>8} {'delta':>8}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in subset:
        da = f"{r.dead_pct_a:.2%}" if not np.isnan(r.dead_pct_a) else "n/a"
        db = f"{r.dead_pct_b:.2%}" if not np.isnan(r.dead_pct_b) else "n/a"
        dd = f"{r.dead_delta:+.2%}" if not np.isnan(r.dead_delta) else "n/a"
        lines.append(f"{r.key:<40} {da:>8} {db:>8} {dd:>8}")
    return "\n".join(lines)
