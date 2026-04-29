"""Signal-to-noise ratio analysis for tensor diffs."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class SNRRow:
    key: str
    snr_a: float  # dB
    snr_b: float  # dB
    snr_delta: float  # snr_b - snr_a
    status: str


def _snr_db(arr: Optional[np.ndarray]) -> float:
    """Compute SNR in dB as 20*log10(mean / std). Returns NaN on failure."""
    if arr is None or arr.size == 0:
        return float("nan")
    flat = arr.astype(float).ravel()
    std = float(np.std(flat))
    if std == 0.0:
        return float("inf") if np.any(flat != 0) else float("nan")
    mean_abs = float(np.mean(np.abs(flat)))
    if mean_abs == 0.0:
        return float("-inf")
    return 20.0 * math.log10(mean_abs / std)


def compute_snr(diff: CheckpointDiff) -> List[SNRRow]:
    """Compute per-key SNR rows for changed and added/removed tensors."""
    rows: List[SNRRow] = []
    for key, td in diff.items():
        if td.status == "unchanged":
            continue
        snr_a = _snr_db(td.array_a)
        snr_b = _snr_db(td.array_b)
        if not math.isnan(snr_a) and not math.isnan(snr_b):
            delta = snr_b - snr_a
        else:
            delta = float("nan")
        rows.append(SNRRow(key=key, snr_a=snr_a, snr_b=snr_b, snr_delta=delta, status=td.status))
    rows.sort(key=lambda r: abs(r.snr_delta) if not math.isnan(r.snr_delta) else -1, reverse=True)
    return rows


def _fmt(v: float, decimals: int = 2) -> str:
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "+inf" if v > 0 else "-inf"
    return f"{v:.{decimals}f}"


def format_snr(rows: List[SNRRow], top_n: Optional[int] = None) -> str:
    """Format SNR rows as a plain-text table."""
    if not rows:
        return "No SNR data available."
    display = rows[:top_n] if top_n else rows
    header = f"{'Key':<40} {'SNR_A (dB)':>12} {'SNR_B (dB)':>12} {'Delta (dB)':>12} {'Status':<10}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in display:
        lines.append(
            f"{r.key:<40} {_fmt(r.snr_a):>12} {_fmt(r.snr_b):>12} {_fmt(r.snr_delta):>12} {r.status:<10}"
        )
    return "\n".join(lines)
