"""Spectral energy analysis: fraction of variance explained by top singular values."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class SpectralRow:
    key: str
    top_k: int
    energy_a: float   # fraction of total variance in A captured by top-k SVs
    energy_b: float   # fraction of total variance in B captured by top-k SVs
    delta: float      # energy_b - energy_a


def _spectral_energy(arr: Optional[np.ndarray], k: int) -> float:
    """Return fraction of variance explained by the top-k singular values."""
    if arr is None:
        return math.nan
    flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(1, -1)
    if flat.size == 0 or flat.shape[0] == 0 or flat.shape[1] == 0:
        return math.nan
    try:
        sv = np.linalg.svd(flat, compute_uv=False)
    except np.linalg.LinAlgError:
        return math.nan
    total = float(np.sum(sv ** 2))
    if total == 0.0:
        return math.nan
    top = int(min(k, len(sv)))
    return float(np.sum(sv[:top] ** 2) / total)


def compute_spectral(
    diff: CheckpointDiff,
    top_k: int = 5,
    include_unchanged: bool = False,
) -> List[SpectralRow]:
    rows: List[SpectralRow] = []
    for key, td in diff.tensors.items():
        if td.status == "removed":
            continue
        if td.status == "unchanged" and not include_unchanged:
            continue
        ea = _spectral_energy(td.a, top_k)
        eb = _spectral_energy(td.b, top_k)
        delta = (eb - ea) if not (math.isnan(ea) or math.isnan(eb)) else math.nan
        rows.append(SpectralRow(key=key, top_k=top_k, energy_a=ea, energy_b=eb, delta=delta))
    rows.sort(key=lambda r: abs(r.delta) if not math.isnan(r.delta) else -1.0, reverse=True)
    return rows


def _fmt(v: float) -> str:
    return "nan" if math.isnan(v) else f"{v:.4f}"


def format_spectral(rows: List[SpectralRow]) -> str:
    if not rows:
        return "No spectral data available."
    header = f"{'key':<40} {'k':>4} {'energy_a':>10} {'energy_b':>10} {'delta':>10}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{r.key:<40} {r.top_k:>4} {_fmt(r.energy_a):>10} {_fmt(r.energy_b):>10} {_fmt(r.delta):>10}"
        )
    return "\n".join(lines)
