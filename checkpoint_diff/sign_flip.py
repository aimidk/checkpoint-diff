"""Detect sign flips in tensor weights between checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class SignFlipRow:
    key: str
    total_elements: int
    flipped: int
    flip_rate: float  # fraction of elements that changed sign
    a_pos_frac: float  # fraction positive in A
    b_pos_frac: float  # fraction positive in B


def _flip_count(a: np.ndarray, b: np.ndarray) -> int:
    """Count elements where sign changed (ignoring zeros)."""
    sign_a = np.sign(a.astype(float))
    sign_b = np.sign(b.astype(float))
    # Only count where both were non-zero and sign differs
    both_nonzero = (sign_a != 0) & (sign_b != 0)
    return int(np.sum(both_nonzero & (sign_a != sign_b)))


def _pos_frac(arr: Optional[np.ndarray]) -> float:
    if arr is None or arr.size == 0:
        return float("nan")
    return float(np.mean(arr.astype(float) > 0))


def compute_sign_flips(
    diff: CheckpointDiff,
    min_flip_rate: float = 0.0,
) -> List[SignFlipRow]:
    """Compute sign-flip statistics for changed tensors."""
    rows: List[SignFlipRow] = []
    for key, td in diff.items():
        if td.status not in ("changed", "added") or td.a is None or td.b is None:
            continue
        if td.a.shape != td.b.shape:
            continue
        total = td.b.size
        flipped = _flip_count(td.a, td.b)
        rate = flipped / total if total > 0 else 0.0
        if rate < min_flip_rate:
            continue
        rows.append(
            SignFlipRow(
                key=key,
                total_elements=total,
                flipped=flipped,
                flip_rate=rate,
                a_pos_frac=_pos_frac(td.a),
                b_pos_frac=_pos_frac(td.b),
            )
        )
    rows.sort(key=lambda r: r.flip_rate, reverse=True)
    return rows


def format_sign_flips(rows: List[SignFlipRow], top_n: Optional[int] = None) -> str:
    if not rows:
        return "No sign flips detected."
    subset = rows[:top_n] if top_n else rows
    header = f"{'Key':<40} {'Flipped':>8} {'Total':>8} {'Rate':>7} {'A+%':>6} {'B+%':>6}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in subset:
        lines.append(
            f"{r.key:<40} {r.flipped:>8} {r.total_elements:>8} "
            f"{r.flip_rate:>6.1%} {r.a_pos_frac:>5.1%} {r.b_pos_frac:>5.1%}"
        )
    return "\n".join(lines)
