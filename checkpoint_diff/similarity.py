"""Cosine and norm-based similarity metrics between checkpoint tensors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .diff import CheckpointDiff


@dataclass
class TensorSimilarity:
    key: str
    cosine: Optional[float]
    l2_norm_a: float
    l2_norm_b: float
    l2_norm_delta: float


SimilarityReport = Dict[str, TensorSimilarity]


def _cosine(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    a_flat = a.flatten().astype(float)
    b_flat = b.flatten().astype(float)
    if a_flat.shape != b_flat.shape:
        return None
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if denom == 0.0:
        return None
    return float(np.dot(a_flat, b_flat) / denom)


def compute_similarity(diff: CheckpointDiff) -> SimilarityReport:
    """Return similarity metrics for every *changed* tensor in the diff."""
    report: SimilarityReport = {}
    for key, td in diff.items():
        if td.array_a is None or td.array_b is None:
            continue
        cosine = _cosine(td.array_a, td.array_b)
        l2_a = float(np.linalg.norm(td.array_a.flatten().astype(float)))
        l2_b = float(np.linalg.norm(td.array_b.flatten().astype(float)))
        report[key] = TensorSimilarity(
            key=key,
            cosine=cosine,
            l2_norm_a=l2_a,
            l2_norm_b=l2_b,
            l2_norm_delta=abs(l2_b - l2_a),
        )
    return report


def format_similarity(report: SimilarityReport) -> str:
    if not report:
        return "No changed tensors to compare."
    lines = [f"{'Key':<40} {'Cosine':>8} {'L2(a)':>10} {'L2(b)':>10} {'|ΔL2|':>10}"]
    lines.append("-" * 82)
    for ts in report.values():
        cos_str = f"{ts.cosine:.6f}" if ts.cosine is not None else "    N/A"
        lines.append(
            f"{ts.key:<40} {cos_str:>8} {ts.l2_norm_a:>10.4f} {ts.l2_norm_b:>10.4f} {ts.l2_norm_delta:>10.4f}"
        )
    return "\n".join(lines)
