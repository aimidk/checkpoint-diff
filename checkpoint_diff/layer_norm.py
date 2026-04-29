"""Per-layer norm analysis: compute L1/L2 norms for each tensor in both checkpoints."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class LayerNormRow:
    key: str
    status: str
    l1_a: float
    l1_b: float
    l2_a: float
    l2_b: float
    l1_delta: float   # l1_b - l1_a
    l2_delta: float   # l2_b - l2_a


def _l1(arr: Optional[np.ndarray]) -> float:
    if arr is None or arr.size == 0:
        return math.nan
    return float(np.sum(np.abs(arr)))


def _l2(arr: Optional[np.ndarray]) -> float:
    if arr is None or arr.size == 0:
        return math.nan
    return float(np.sqrt(np.sum(arr.astype(float) ** 2)))


def compute_layer_norms(diff: CheckpointDiff) -> List[LayerNormRow]:
    """Return a LayerNormRow for every tensor in the diff."""
    rows: List[LayerNormRow] = []
    for key, td in diff.items():
        la = _l1(td.array_a)
        lb = _l1(td.array_b)
        ra = _l2(td.array_a)
        rb = _l2(td.array_b)
        rows.append(
            LayerNormRow(
                key=key,
                status=td.status,
                l1_a=la,
                l1_b=lb,
                l2_a=ra,
                l2_b=rb,
                l1_delta=lb - la if not (math.isnan(la) or math.isnan(lb)) else math.nan,
                l2_delta=rb - ra if not (math.isnan(ra) or math.isnan(rb)) else math.nan,
            )
        )
    rows.sort(key=lambda r: abs(r.l2_delta) if not math.isnan(r.l2_delta) else 0.0, reverse=True)
    return rows


def _fmt(v: float) -> str:
    return "nan" if math.isnan(v) else f"{v:.6g}"


def format_layer_norms(rows: List[LayerNormRow], top_n: Optional[int] = None) -> str:
    """Return a human-readable table of layer norms."""
    if not rows:
        return "No layer norm data."
    displayed = rows[:top_n] if top_n else rows
    header = f"{'Key':<40} {'Status':<10} {'L1(a)':>12} {'L1(b)':>12} {'L2(a)':>12} {'L2(b)':>12} {'ΔL2':>12}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in displayed:
        lines.append(
            f"{r.key:<40} {r.status:<10} {_fmt(r.l1_a):>12} {_fmt(r.l1_b):>12}"
            f" {_fmt(r.l2_a):>12} {_fmt(r.l2_b):>12} {_fmt(r.l2_delta):>12}"
        )
    return "\n".join(lines)
