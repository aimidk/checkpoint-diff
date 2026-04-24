"""Percentile-based analysis of tensor weight distributions across checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


@dataclass
class PercentileRow:
    key: str
    percentiles_a: Dict[int, float]
    percentiles_b: Dict[int, float]
    deltas: Dict[int, float]


def _compute_percentiles(arr: np.ndarray) -> Dict[int, float]:
    flat = arr.flatten().astype(float)
    return {p: float(np.percentile(flat, p)) for p in PERCENTILES}


def compute_percentiles(
    diff: CheckpointDiff,
    keys: Optional[List[str]] = None,
) -> List[PercentileRow]:
    """Compute per-percentile statistics for changed/added tensors."""
    rows: List[PercentileRow] = []
    target_keys = keys if keys is not None else list(diff.keys())

    for key in target_keys:
        td: TensorDiff = diff[key]
        if td.status == "removed":
            continue
        if td.tensor_a is None and td.tensor_b is None:
            continue

        pa = _compute_percentiles(td.tensor_a) if td.tensor_a is not None else {p: float("nan") for p in PERCENTILES}
        pb = _compute_percentiles(td.tensor_b) if td.tensor_b is not None else {p: float("nan") for p in PERCENTILES}
        deltas = {
            p: (pb[p] - pa[p]) if not (np.isnan(pa[p]) or np.isnan(pb[p])) else float("nan")
            for p in PERCENTILES
        }
        rows.append(PercentileRow(key=key, percentiles_a=pa, percentiles_b=pb, deltas=deltas))

    return rows


def format_percentiles(rows: List[PercentileRow], show_delta: bool = True) -> str:
    """Render percentile rows as a human-readable table."""
    if not rows:
        return "No percentile data available."

    header_cols = [f"p{p:02d}" for p in PERCENTILES]
    col_w = 10
    key_w = max(len(r.key) for r in rows)

    header = f"{'key':<{key_w}}  " + "  ".join(f"{h:>{col_w}}" for h in header_cols)
    sep = "-" * len(header)
    lines = [header, sep]

    for row in rows:
        src = row.deltas if show_delta else row.percentiles_b
        vals = "  ".join(f"{src[p]:>{col_w}.4f}" for p in PERCENTILES)
        label = "Δ" if show_delta else "b"
        lines.append(f"{row.key:<{key_w}}  {vals}  [{label}]")

    return "\n".join(lines)
