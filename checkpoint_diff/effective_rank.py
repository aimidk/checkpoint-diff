"""Effective rank analysis for tensor weight matrices."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class EffectiveRankRow:
    key: str
    rank_a: float
    rank_b: float
    delta: float
    status: str


def _effective_rank(arr: Optional[np.ndarray]) -> float:
    """Compute effective rank via singular value entropy (Roy & Vetterli 2007)."""
    if arr is None or arr.size == 0:
        return float("nan")
    flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(1, -1)
    try:
        sv = np.linalg.svd(flat, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    sv = sv[sv > 0]
    if sv.size == 0:
        return float("nan")
    p = sv / sv.sum()
    entropy = -float(np.sum(p * np.log(p)))
    return math.exp(entropy)


def compute_effective_rank(diff: CheckpointDiff) -> list[EffectiveRankRow]:
    rows: list[EffectiveRankRow] = []
    for key, td in diff.items():
        if td.status == "removed":
            continue
        rank_a = _effective_rank(td.array_a)
        rank_b = _effective_rank(td.array_b)
        delta = (rank_b - rank_a) if not (math.isnan(rank_a) or math.isnan(rank_b)) else float("nan")
        rows.append(EffectiveRankRow(key=key, rank_a=rank_a, rank_b=rank_b, delta=delta, status=td.status))
    rows.sort(key=lambda r: abs(r.delta) if not math.isnan(r.delta) else 0.0, reverse=True)
    return rows


def _fmt(v: float, decimals: int = 3) -> str:
    return f"{v:.{decimals}f}" if not math.isnan(v) else "nan"


def format_effective_rank(rows: list[EffectiveRankRow], top_n: Optional[int] = None) -> str:
    if not rows:
        return "No effective rank data."
    shown = rows[:top_n] if top_n else rows
    header = f"{'Key':<40} {'RankA':>8} {'RankB':>8} {'Delta':>8} {'Status':<10}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in shown:
        lines.append(
            f"{r.key:<40} {_fmt(r.rank_a):>8} {_fmt(r.rank_b):>8} {_fmt(r.delta):>8} {r.status:<10}"
        )
    return "\n".join(lines)
