"""Detect outlier tensors based on statistical thresholds."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .diff import CheckpointDiff, TensorDiff


@dataclass
class OutlierResult:
    key: str
    mean: float
    std: float
    abs_max: float
    reason: str


def _stats(td: TensorDiff) -> tuple[float, float, float]:
    a = td.tensor_b if td.tensor_b is not None else td.tensor_a
    flat = a.astype(np.float64).ravel()
    return float(np.mean(flat)), float(np.std(flat)), float(np.max(np.abs(flat)))


def detect_outliers(
    diff: CheckpointDiff,
    *,
    max_abs_mean: float | None = None,
    max_std: float | None = None,
    max_abs_max: float | None = None,
    statuses: tuple[str, ...] = ("changed", "added"),
) -> List[OutlierResult]:
    results: List[OutlierResult] = []
    for key, td in diff.items():
        if td.status not in statuses:
            continue
        mean, std, abs_max = _stats(td)
        reasons = []
        if max_abs_mean is not None and abs(mean) > max_abs_mean:
            reasons.append(f"|mean|={abs(mean):.4g} > {max_abs_mean}")
        if max_std is not None and std > max_std:
            reasons.append(f"std={std:.4g} > {max_std}")
        if max_abs_max is not None and abs_max > max_abs_max:
            reasons.append(f"abs_max={abs_max:.4g} > {max_abs_max}")
        if reasons:
            results.append(OutlierResult(key=key, mean=mean, std=std, abs_max=abs_max, reason="; ".join(reasons)))
    return results


def format_outliers(results: List[OutlierResult]) -> str:
    if not results:
        return "No outliers detected."
    lines = [f"{'Key':<40} {'Mean':>12} {'Std':>12} {'AbsMax':>12}  Reason"]
    lines.append("-" * 100)
    for r in results:
        lines.append(f"{r.key:<40} {r.mean:>12.4g} {r.std:>12.4g} {r.abs_max:>12.4g}  {r.reason}")
    return "\n".join(lines)
