"""Quantile shift analysis between two checkpoint tensors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff

_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


@dataclass
class QuantileShiftRow:
    key: str
    quantiles: tuple  # (0.1, 0.25, 0.5, 0.75, 0.9)
    shift_a: List[float]  # quantile values in checkpoint A
    shift_b: List[float]  # quantile values in checkpoint B
    max_abs_shift: float  # max absolute difference across quantiles


def _quantile_values(arr: Optional[np.ndarray]) -> List[float]:
    if arr is None or arr.size == 0:
        return [float("nan")] * len(_QUANTILES)
    flat = arr.flatten().astype(float)
    return [float(np.quantile(flat, q)) for q in _QUANTILES]


def compute_quantile_shifts(
    diff: CheckpointDiff,
    *,
    include_unchanged: bool = False,
    top_n: Optional[int] = None,
) -> List[QuantileShiftRow]:
    """Compute per-key quantile shifts for changed/added keys."""
    rows: List[QuantileShiftRow] = []

    for key, td in diff.items():
        if td.status == "removed":
            continue
        if td.status == "unchanged" and not include_unchanged:
            continue

        qa = _quantile_values(td.array_a)
        qb = _quantile_values(td.array_b)

        valid_pairs = [
            (a, b) for a, b in zip(qa, qb)
            if not (np.isnan(a) or np.isnan(b))
        ]
        max_shift = max((abs(b - a) for a, b in valid_pairs), default=float("nan"))

        rows.append(
            QuantileShiftRow(
                key=key,
                quantiles=_QUANTILES,
                shift_a=qa,
                shift_b=qb,
                max_abs_shift=max_shift,
            )
        )

    rows.sort(key=lambda r: r.max_abs_shift if not np.isnan(r.max_abs_shift) else -1, reverse=True)
    if top_n is not None:
        rows = rows[:top_n]
    return rows


def format_quantile_shifts(rows: List[QuantileShiftRow]) -> str:
    if not rows:
        return "No quantile shift data."

    q_header = "  ".join(f"q{int(q*100):02d}" for q in _QUANTILES)
    header = f"{'Key':<40}  {'A: ' + q_header:<50}  {'B: ' + q_header:<50}  MaxShift"
    lines = [header, "-" * len(header)]

    def _fmt(vals: List[float]) -> str:
        return "  ".join(f"{v:+.3f}" if not np.isnan(v) else "  nan " for v in vals)

    for row in rows:
        shift_str = f"{row.max_abs_shift:.4f}" if not np.isnan(row.max_abs_shift) else "nan"
        lines.append(f"{row.key:<40}  {_fmt(row.shift_a):<50}  {_fmt(row.shift_b):<50}  {shift_str}")

    return "\n".join(lines)
