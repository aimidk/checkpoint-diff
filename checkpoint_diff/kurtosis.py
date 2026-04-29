"""Kurtosis analysis for tensor weight distributions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class KurtosisRow:
    key: str
    kurtosis_a: float
    kurtosis_b: float
    delta: float
    status: str


def _kurtosis(arr: Optional[np.ndarray]) -> float:
    """Compute excess kurtosis (Fisher's definition) for a flat array."""
    if arr is None:
        return float("nan")
    flat = arr.flatten().astype(float)
    if flat.size < 4:
        return float("nan")
    mu = flat.mean()
    sigma = flat.std()
    if sigma == 0.0:
        return float("nan")
    return float(np.mean(((flat - mu) / sigma) ** 4) - 3.0)


def compute_kurtosis(
    diff: CheckpointDiff,
    *,
    include_unchanged: bool = False,
    top_n: Optional[int] = None,
) -> List[KurtosisRow]:
    """Return kurtosis rows for each tensor in *diff*."""
    rows: List[KurtosisRow] = []
    for key, td in diff.tensors.items():
        if td.status == "removed":
            continue
        if td.status == "unchanged" and not include_unchanged:
            continue
        ka = _kurtosis(td.array_a)
        kb = _kurtosis(td.array_b)
        delta = (kb - ka) if not (np.isnan(ka) or np.isnan(kb)) else float("nan")
        rows.append(KurtosisRow(key=key, kurtosis_a=ka, kurtosis_b=kb, delta=delta, status=td.status))

    rows.sort(key=lambda r: abs(r.delta) if not np.isnan(r.delta) else 0.0, reverse=True)
    if top_n is not None:
        rows = rows[:top_n]
    return rows


def format_kurtosis(rows: List[KurtosisRow]) -> str:
    """Render kurtosis rows as a plain-text table."""
    if not rows:
        return "No kurtosis data to display."
    header = f"{'Key':<40} {'Kurt-A':>10} {'Kurt-B':>10} {'Delta':>10} {'Status':<10}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        ka = f"{r.kurtosis_a:10.4f}" if not np.isnan(r.kurtosis_a) else f"{'nan':>10}"
        kb = f"{r.kurtosis_b:10.4f}" if not np.isnan(r.kurtosis_b) else f"{'nan':>10}"
        d = f"{r.delta:10.4f}" if not np.isnan(r.delta) else f"{'nan':>10}"
        lines.append(f"{r.key:<40} {ka} {kb} {d} {r.status:<10}")
    return "\n".join(lines)
