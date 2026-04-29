"""Compute per-key L2 norm ratios between checkpoint A and B."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .diff import CheckpointDiff


@dataclass
class NormRatioRow:
    key: str
    norm_a: float
    norm_b: float
    ratio: float  # norm_b / norm_a; nan when norm_a == 0
    status: str


def _l2(arr: Optional[np.ndarray]) -> float:
    if arr is None or arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.sum(arr.astype(float) ** 2)))


def _ratio(norm_a: float, norm_b: float) -> float:
    if math.isnan(norm_a) or math.isnan(norm_b):
        return float("nan")
    if norm_a == 0.0:
        return float("nan")
    return norm_b / norm_a


def compute_norm_ratios(
    diff: CheckpointDiff,
    *,
    include_unchanged: bool = False,
) -> List[NormRatioRow]:
    rows: List[NormRatioRow] = []
    for key, td in diff.tensors.items():
        if td.status == "unchanged" and not include_unchanged:
            continue
        na = _l2(td.data_a)
        nb = _l2(td.data_b)
        rows.append(
            NormRatioRow(
                key=key,
                norm_a=na,
                norm_b=nb,
                ratio=_ratio(na, nb),
                status=td.status,
            )
        )
    rows.sort(key=lambda r: abs(r.ratio - 1.0) if not math.isnan(r.ratio) else float("inf"), reverse=True)
    return rows


def _fmt(v: float, precision: int = 4) -> str:
    return "nan" if math.isnan(v) else f"{v:.{precision}f}"


def format_norm_ratios(rows: List[NormRatioRow], *, top_n: Optional[int] = None) -> str:
    if not rows:
        return "No norm ratio data."
    shown = rows[:top_n] if top_n else rows
    header = f"{'Key':<40} {'norm_a':>12} {'norm_b':>12} {'ratio':>10} {'status':<10}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in shown:
        lines.append(
            f"{r.key:<40} {_fmt(r.norm_a):>12} {_fmt(r.norm_b):>12} {_fmt(r.ratio):>10} {r.status:<10}"
        )
    return "\n".join(lines)
