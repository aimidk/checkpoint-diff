"""Detect weight symmetry between corresponding tensor pairs in a checkpoint diff."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class SymmetryRow:
    key: str
    mean_a: float
    mean_b: float
    mean_symmetry: float   # 1 - |mean_a + mean_b| / (|mean_a| + |mean_b| + eps)
    std_a: float
    std_b: float
    std_ratio: float       # min/max of (std_a, std_b); 1.0 = perfectly matched
    is_symmetric: bool


_EPS = 1e-9


def _mean_symmetry(a: float, b: float) -> float:
    """Score in [0, 1]: how close mean_a and mean_b are to being negatives of each other."""
    numerator = abs(a + b)
    denominator = abs(a) + abs(b) + _EPS
    return float(1.0 - numerator / denominator)


def _std_ratio(a: float, b: float) -> float:
    """Ratio of min to max std; 1.0 means identical spread."""
    lo, hi = sorted([abs(a), abs(b)])
    return float(lo / (hi + _EPS))


def compute_symmetry(
    diff: CheckpointDiff,
    mean_threshold: float = 0.9,
    std_threshold: float = 0.9,
) -> List[SymmetryRow]:
    """Return a SymmetryRow for each changed tensor in *diff*."""
    rows: List[SymmetryRow] = []
    for key, td in diff.items():
        if td.status not in ("changed", "added"):
            continue
        if td.mean_a is None or td.mean_b is None:
            continue
        ms = _mean_symmetry(td.mean_a, td.mean_b)
        sr = _std_ratio(td.std_a or 0.0, td.std_b or 0.0)
        rows.append(
            SymmetryRow(
                key=key,
                mean_a=td.mean_a,
                mean_b=td.mean_b,
                mean_symmetry=ms,
                std_a=td.std_a or float("nan"),
                std_b=td.std_b or float("nan"),
                std_ratio=sr,
                is_symmetric=(ms >= mean_threshold and sr >= std_threshold),
            )
        )
    rows.sort(key=lambda r: r.mean_symmetry, reverse=True)
    return rows


def format_symmetry(rows: List[SymmetryRow]) -> str:
    if not rows:
        return "No symmetry data available."
    header = f"{'Key':<40} {'MeanSym':>8} {'StdRatio':>9} {'Symmetric':>10}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        sym_flag = "yes" if r.is_symmetric else "no"
        lines.append(
            f"{r.key:<40} {r.mean_symmetry:>8.4f} {r.std_ratio:>9.4f} {sym_flag:>10}"
        )
    return "\n".join(lines)
