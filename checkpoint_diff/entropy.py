"""Compute weight entropy metrics for tensor diffs."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class EntropyRow:
    key: str
    entropy_a: float
    entropy_b: float
    delta: float


def _histogram_entropy(arr: np.ndarray, bins: int = 64) -> float:
    """Approximate Shannon entropy via histogram binning."""
    flat = arr.astype(float).ravel()
    if flat.size == 0:
        return float("nan")
    counts, _ = np.histogram(flat, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_entropy(
    diff: CheckpointDiff,
    bins: int = 64,
    include_unchanged: bool = False,
) -> List[EntropyRow]:
    """Return entropy rows for each tensor key in *diff*.

    Parameters
    ----------
    diff:
        The checkpoint diff to analyse.
    bins:
        Number of histogram bins used when approximating entropy.
    include_unchanged:
        When *True*, unchanged tensors are included in the output.
    """
    rows: List[EntropyRow] = []
    for key, td in diff.items():
        if td.status == "removed":
            continue
        if td.status == "unchanged" and not include_unchanged:
            continue

        if td.tensor_a is not None:
            ent_a = _histogram_entropy(td.tensor_a, bins=bins)
        else:
            ent_a = float("nan")

        if td.tensor_b is not None:
            ent_b = _histogram_entropy(td.tensor_b, bins=bins)
        else:
            ent_b = float("nan")

        if math.isnan(ent_a) or math.isnan(ent_b):
            delta = float("nan")
        else:
            delta = ent_b - ent_a

        rows.append(EntropyRow(key=key, entropy_a=ent_a, entropy_b=ent_b, delta=delta))

    rows.sort(key=lambda r: abs(r.delta) if not math.isnan(r.delta) else 0.0, reverse=True)
    return rows


def format_entropy(rows: List[EntropyRow], top_n: Optional[int] = None) -> str:
    """Render entropy rows as a plain-text table."""
    if not rows:
        return "No entropy data."
    if top_n is not None:
        rows = rows[:top_n]
    header = f"{'Key':<40} {'H(A)':>8} {'H(B)':>8} {'Delta':>8}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        ha = f"{r.entropy_a:.4f}" if not math.isnan(r.entropy_a) else "  n/a"
        hb = f"{r.entropy_b:.4f}" if not math.isnan(r.entropy_b) else "  n/a"
        d = f"{r.delta:+.4f}" if not math.isnan(r.delta) else "  n/a"
        lines.append(f"{r.key:<40} {ha:>8} {hb:>8} {d:>8}")
    return "\n".join(lines)
