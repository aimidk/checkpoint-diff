"""Activation statistics: fraction of positive, negative, and zero activations per tensor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import math
import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class ActivationRow:
    key: str
    pos_frac_a: float   # fraction > 0 in checkpoint A
    pos_frac_b: float   # fraction > 0 in checkpoint B
    neg_frac_a: float
    neg_frac_b: float
    zero_frac_a: float
    zero_frac_b: float
    pos_delta: float    # pos_frac_b - pos_frac_a


def _fractions(arr: Optional[np.ndarray]):
    """Return (pos, neg, zero) fractions for *arr*, or (nan, nan, nan) if None."""
    if arr is None or arr.size == 0:
        nan = float("nan")
        return nan, nan, nan
    flat = arr.flatten().astype(float)
    n = flat.size
    pos = float(np.sum(flat > 0)) / n
    neg = float(np.sum(flat < 0)) / n
    zero = float(np.sum(flat == 0)) / n
    return pos, neg, zero


def compute_activations(
    diff: CheckpointDiff,
    *,
    include_unchanged: bool = False,
    top_n: Optional[int] = None,
) -> List[ActivationRow]:
    rows: List[ActivationRow] = []
    for key, td in diff.tensors.items():
        if td.status == "removed":
            continue
        if td.status == "unchanged" and not include_unchanged:
            continue
        pa, na, za = _fractions(td.array_a)
        pb, nb, zb = _fractions(td.array_b)
        delta = (pb - pa) if not (math.isnan(pb) or math.isnan(pa)) else float("nan")
        rows.append(ActivationRow(
            key=key,
            pos_frac_a=pa, pos_frac_b=pb,
            neg_frac_a=na, neg_frac_b=nb,
            zero_frac_a=za, zero_frac_b=zb,
            pos_delta=delta,
        ))
    rows.sort(key=lambda r: abs(r.pos_delta) if not math.isnan(r.pos_delta) else 0.0, reverse=True)
    if top_n is not None:
        rows = rows[:top_n]
    return rows


def format_activations(rows: List[ActivationRow]) -> str:
    if not rows:
        return "No activation data."
    header = f"{'Key':<40} {'pos_a':>7} {'pos_b':>7} {'neg_a':>7} {'neg_b':>7} {'zero_a':>7} {'zero_b':>7} {'Δpos':>8}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        def _f(v: float) -> str:
            return "nan" if math.isnan(v) else f"{v:.4f}"
        lines.append(
            f"{r.key:<40} {_f(r.pos_frac_a):>7} {_f(r.pos_frac_b):>7} "
            f"{_f(r.neg_frac_a):>7} {_f(r.neg_frac_b):>7} "
            f"{_f(r.zero_frac_a):>7} {_f(r.zero_frac_b):>7} {_f(r.pos_delta):>8}"
        )
    return "\n".join(lines)
