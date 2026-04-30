"""Fréchet distance between tensor distributions in two checkpoints."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class FrechetRow:
    key: str
    frechet: float  # Fréchet distance between A and B distributions
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float


def _frechet(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Compute 1-D Fréchet distance (Wasserstein-2 for Gaussians).

    For Gaussian distributions N(mu_a, sigma_a) and N(mu_b, sigma_b):
        FD = (mu_a - mu_b)^2 + (sigma_a - sigma_b)^2
    """
    if a is None or b is None:
        return float("nan")
    flat_a = a.astype(float).ravel()
    flat_b = b.astype(float).ravel()
    if flat_a.size == 0 or flat_b.size == 0:
        return float("nan")
    mu_a, sigma_a = float(np.mean(flat_a)), float(np.std(flat_a))
    mu_b, sigma_b = float(np.mean(flat_b)), float(np.std(flat_b))
    return (mu_a - mu_b) ** 2 + (sigma_a - sigma_b) ** 2


def compute_frechet(diff: CheckpointDiff, top_n: Optional[int] = None) -> List[FrechetRow]:
    """Return Fréchet distance rows for all changed/added keys."""
    rows: List[FrechetRow] = []
    for key, td in diff.tensors.items():
        if td.status == "removed":
            continue
        a = td.array_a
        b = td.array_b
        fd = _frechet(a, b)
        flat_a = a.astype(float).ravel() if a is not None else np.array([])
        flat_b = b.astype(float).ravel() if b is not None else np.array([])
        rows.append(
            FrechetRow(
                key=key,
                frechet=fd,
                mean_a=float(np.mean(flat_a)) if flat_a.size else float("nan"),
                mean_b=float(np.mean(flat_b)) if flat_b.size else float("nan"),
                std_a=float(np.std(flat_a)) if flat_a.size else float("nan"),
                std_b=float(np.std(flat_b)) if flat_b.size else float("nan"),
            )
        )
    rows.sort(key=lambda r: (math.isnan(r.frechet), -r.frechet if not math.isnan(r.frechet) else 0))
    if top_n is not None:
        rows = rows[:top_n]
    return rows


def _fmt(v: float) -> str:
    return "nan" if math.isnan(v) else f"{v:.6f}"


def format_frechet(rows: List[FrechetRow]) -> str:
    """Return a human-readable table of Fréchet distances."""
    if not rows:
        return "No Fréchet distance data available."
    header = f"{'Key':<40} {'FD':>12} {'mean_a':>10} {'mean_b':>10} {'std_a':>10} {'std_b':>10}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{r.key:<40} {_fmt(r.frechet):>12} {_fmt(r.mean_a):>10} "
            f"{_fmt(r.mean_b):>10} {_fmt(r.std_a):>10} {_fmt(r.std_b):>10}"
        )
    return "\n".join(lines)
