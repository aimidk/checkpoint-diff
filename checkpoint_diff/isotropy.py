"""Isotropy analysis: measures how uniformly distributed weight directions are.

A perfectly isotropic tensor has equal variance in all directions; low isotropy
indicates the weights are concentrated along a few axes (often a sign of
collapse or poor initialization).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class IsotropyRow:
    key: str
    isotropy_a: float   # [0, 1] – 1.0 means perfectly isotropic
    isotropy_b: float
    delta: float        # isotropy_b - isotropy_a


def _isotropy(arr: Optional[np.ndarray]) -> float:
    """Compute isotropy score via singular-value entropy.

    Isotropy = exp(H) / rank  where H is the Shannon entropy of the
    normalised squared singular values.  Returns NaN for degenerate inputs.
    """
    if arr is None or arr.size == 0:
        return math.nan
    flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(-1, 1)
    if flat.shape[0] < 2 or flat.shape[1] < 1:
        return math.nan
    try:
        sv = np.linalg.svd(flat, compute_uv=False)
    except np.linalg.LinAlgError:
        return math.nan
    sv2 = sv ** 2
    total = sv2.sum()
    if total == 0:
        return math.nan
    p = sv2 / total
    # clip to avoid log(0)
    p = np.clip(p, 1e-12, None)
    entropy = -float(np.sum(p * np.log(p)))
    rank = len(sv)
    return math.exp(entropy) / rank


def compute_isotropy(
    diff: CheckpointDiff,
    *,
    top_n: Optional[int] = None,
) -> List[IsotropyRow]:
    """Return an isotropy row for every changed/added tensor in *diff*."""
    rows: List[IsotropyRow] = []
    for key, td in diff.items():
        if td.status == "removed":
            continue
        iso_a = _isotropy(td.array_a)
        iso_b = _isotropy(td.array_b)
        delta = (iso_b - iso_a) if not (math.isnan(iso_a) or math.isnan(iso_b)) else math.nan
        rows.append(IsotropyRow(key=key, isotropy_a=iso_a, isotropy_b=iso_b, delta=delta))
    rows.sort(key=lambda r: abs(r.delta) if not math.isnan(r.delta) else 0.0, reverse=True)
    if top_n is not None:
        rows = rows[:top_n]
    return rows


def _fmt(v: float) -> str:
    return "nan" if math.isnan(v) else f"{v:.4f}"


def format_isotropy(rows: List[IsotropyRow]) -> str:
    if not rows:
        return "No isotropy data."
    header = f"{'Key':<40} {'Iso-A':>8} {'Iso-B':>8} {'Delta':>8}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{r.key:<40} {_fmt(r.isotropy_a):>8} {_fmt(r.isotropy_b):>8} {_fmt(r.delta):>8}"
        )
    return "\n".join(lines)
