"""Compute pairwise weight correlation between two checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class CorrelationRow:
    key: str
    pearson: float  # -1 … 1, or NaN when undefined
    status: str


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson r for two flat arrays; NaN when std is zero."""
    a = a.astype(float).ravel()
    b = b.astype(float).ravel()
    if a.size != b.size or a.size < 2:
        return float("nan")
    std_a, std_b = a.std(), b.std()
    if std_a == 0.0 or std_b == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_correlations(
    diff: CheckpointDiff,
    *,
    include_unchanged: bool = False,
) -> List[CorrelationRow]:
    """Return one CorrelationRow per tensor that has both A and B arrays."""
    rows: List[CorrelationRow] = []
    for key, td in diff.items():
        if td.array_a is None or td.array_b is None:
            continue
        if td.status == "unchanged" and not include_unchanged:
            continue
        rows.append(
            CorrelationRow(
                key=key,
                pearson=_pearson(td.array_a, td.array_b),
                status=td.status,
            )
        )
    rows.sort(key=lambda r: (r.pearson if not np.isnan(r.pearson) else 2.0))
    return rows


def format_correlations(
    rows: List[CorrelationRow],
    *,
    top_n: Optional[int] = None,
) -> str:
    """Return a human-readable table of correlation rows."""
    if not rows:
        return "No correlation data available."
    display = rows[:top_n] if top_n else rows
    header = f"{'Key':<40} {'Pearson r':>10}  Status"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in display:
        pearson_str = f"{r.pearson:>10.4f}" if not np.isnan(r.pearson) else f"{'nan':>10}"
        lines.append(f"{r.key:<40} {pearson_str}  {r.status}")
    return "\n".join(lines)
