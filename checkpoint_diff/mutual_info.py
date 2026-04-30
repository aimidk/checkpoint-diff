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
    mi_bits: float          # mutual information in bits
    entropy_a: float        # marginal entropy of tensor A
    entropy_b: float        # marginal entropy of tensor B
    normalized_mi: float    # NMI in [0, 1]


def _histogram_entropy(arr: np.ndarray, bins: int = 32) -> float:
    """Estimate Shannon entropy (bits) via histogram."""
    if arr is None or arr.size == 0:
        return float("nan")
    flat = arr.flatten().astype(float)
    counts, _ = np.histogram(flat, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _joint_entropy(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """Estimate joint Shannon entropy (bits) via 2-D histogram."""
    fa = a.flatten().astype(float)
    fb = b.flatten().astype(float)
    n = min(len(fa), len(fb))
    counts, _, _ = np.histogram2d(fa[:n], fb[:n], bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _compute_mi(td: TensorDiff, bins: int = 32) -> MutualInfoRow:
    a, b = td.array_a, td.array_b
    ha = _histogram_entropy(a, bins)
    hb = _histogram_entropy(b, bins)
    if a is None or b is None or a.size == 0 or b.size == 0:
        return MutualInfoRow(
            key=td.key,
            mi_bits=float("nan"),
            entropy_a=ha,
            entropy_b=hb,
            normalized_mi=float("nan"),
        )
    hjoint = _joint_entropy(a, b, bins)
    mi = max(0.0, ha + hb - hjoint)
    denom = min(ha, hb)
    nmi = mi / denom if denom > 0 else float("nan")
    return MutualInfoRow(key=td.key, mi_bits=mi, entropy_a=ha, entropy_b=hb, normalized_mi=nmi)


def compute_mutual_info(
    diff: CheckpointDiff,
    bins: int = 32,
    top_n: Optional[int] = None,
) -> list[MutualInfoRow]:
    """Compute mutual information for all changed keys with both tensors present."""
    rows = []
    for td in diff.values():
        if td.array_a is None or td.array_b is None:
            continue
        rows.append(_compute_mi(td, bins=bins))
    rows.sort(key=lambda r: (math.isnan(r.mi_bits), -r.mi_bits if not math.isnan(r.mi_bits) else 0))
    return rows[:top_n] if top_n is not None else rows


def format_mutual_info(rows: list[MutualInfoRow]) -> str:
    if not rows:
        return "No mutual information data available."
    header = f"{'Key':<40} {'MI (bits)':>10} {'H(A)':>8} {'H(B)':>8} {'NMI':>8}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        mi = f"{r.mi_bits:.4f}" if not math.isnan(r.mi_bits) else "nan"
        ha = f"{r.entropy_a:.4f}" if not math.isnan(r.entropy_a) else "nan"
        hb = f"{r.entropy_b:.4f}" if not math.isnan(r.entropy_b) else "nan"
        nmi = f"{r.normalized_mi:.4f}" if not math.isnan(r.normalized_mi) else "nan"
        lines.append(f"{r.key:<40} {mi:>10} {ha:>8} {hb:>8} {nmi:>8}")
    return "\n".join(lines)
