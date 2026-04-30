"""Mutual information estimation between tensor pairs across checkpoints."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class MutualInfoRow:
    key: str
    mi_bits: float  # estimated mutual information in bits
    entropy_a: float
    entropy_b: float
    normalized_mi: float  # MI / max(H(a), H(b)), in [0, 1]


def _histogram_entropy(arr: np.ndarray, bins: int = 64) -> float:
    """Estimate Shannon entropy from histogram (in bits)."""
    if arr is None or arr.size == 0:
        return math.nan
    flat = arr.flatten().astype(float)
    counts, _ = np.histogram(flat, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _joint_mi(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """Estimate MI between two arrays via 2-D joint histogram."""
    if a is None or b is None or a.size == 0 or b.size == 0:
        return math.nan
    fa = a.flatten().astype(float)
    fb = b.flatten().astype(float)
    min_len = min(len(fa), len(fb))
    fa, fb = fa[:min_len], fb[:min_len]
    joint, _, _ = np.histogram2d(fa, fb, bins=bins)
    joint_prob = joint / joint.sum()
    pa = joint_prob.sum(axis=1)
    pb = joint_prob.sum(axis=0)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            p_ij = joint_prob[i, j]
            if p_ij > 0 and pa[i] > 0 and pb[j] > 0:
                mi += p_ij * math.log2(p_ij / (pa[i] * pb[j]))
    return float(mi)


def compute_mutual_info(
    diff: CheckpointDiff,
    bins: int = 64,
    top_n: Optional[int] = None,
) -> list[MutualInfoRow]:
    rows: list[MutualInfoRow] = []
    for key, td in diff.tensors.items():
        if td.array_a is None or td.array_b is None:
            continue
        ha = _histogram_entropy(td.array_a, bins=bins)
        hb = _histogram_entropy(td.array_b, bins=bins)
        mi = _joint_mi(td.array_a, td.array_b, bins=bins)
        denom = max(ha, hb) if (not math.isnan(ha) and not math.isnan(hb)) else math.nan
        nmi = mi / denom if (not math.isnan(denom) and denom > 0) else math.nan
        rows.append(MutualInfoRow(key=key, mi_bits=mi, entropy_a=ha, entropy_b=hb, normalized_mi=nmi))
    rows.sort(key=lambda r: r.mi_bits if not math.isnan(r.mi_bits) else -1, reverse=True)
    if top_n is not None:
        rows = rows[:top_n]
    return rows


def _fmt(v: float) -> str:
    return "nan" if math.isnan(v) else f"{v:.4f}"


def format_mutual_info(rows: list[MutualInfoRow]) -> str:
    if not rows:
        return "No mutual information data available."
    header = f"{'Key':<40} {'MI (bits)':>10} {'H(a)':>8} {'H(b)':>8} {'NMI':>8}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"{r.key:<40} {_fmt(r.mi_bits):>10} {_fmt(r.entropy_a):>8} "
            f"{_fmt(r.entropy_b):>8} {_fmt(r.normalized_mi):>8}"
        )
    return "\n".join(lines)
