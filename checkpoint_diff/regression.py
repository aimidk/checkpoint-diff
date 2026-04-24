"""Regression detection: flag tensors that moved away from a reference baseline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class RegressionResult:
    key: str
    mean_a: float
    mean_b: float
    mean_ref: float
    regressed: bool
    direction: str  # 'toward' | 'away' | 'neutral'


@dataclass
class RegressionReport:
    results: List[RegressionResult] = field(default_factory=list)

    @property
    def flagged(self) -> List[RegressionResult]:
        return [r for r in self.results if r.regressed]


def _safe_mean(arr: np.ndarray) -> float:
    try:
        return float(np.mean(arr))
    except Exception:
        return 0.0


def detect_regression(
    diff: CheckpointDiff,
    reference: Dict[str, np.ndarray],
    tolerance: float = 0.0,
) -> RegressionReport:
    """Compare changed tensors against a reference checkpoint.

    A tensor is *regressed* when the B-side moves *further* from the reference
    mean than the A-side was (by more than *tolerance*).
    """
    report = RegressionReport()
    for key, td in diff.tensors.items():
        if td.status not in ("changed", "added"):
            continue
        if key not in reference:
            continue
        ref_mean = _safe_mean(reference[key])
        mean_a = td.mean_a if td.mean_a is not None else ref_mean
        mean_b = td.mean_b if td.mean_b is not None else ref_mean
        dist_a = abs(mean_a - ref_mean)
        dist_b = abs(mean_b - ref_mean)
        regressed = (dist_b - dist_a) > tolerance
        if dist_b < dist_a - tolerance:
            direction = "toward"
        elif regressed:
            direction = "away"
        else:
            direction = "neutral"
        report.results.append(
            RegressionResult(
                key=key,
                mean_a=mean_a,
                mean_b=mean_b,
                mean_ref=ref_mean,
                regressed=regressed,
                direction=direction,
            )
        )
    return report


def format_regression(report: RegressionReport, show_all: bool = False) -> str:
    rows = report.flagged if not show_all else report.results
    if not rows:
        return "No regressions detected."
    lines = [f"{'Key':<40} {'MeanA':>10} {'MeanB':>10} {'Ref':>10} {'Dir':<8}"]
    lines.append("-" * 82)
    for r in rows:
        lines.append(
            f"{r.key:<40} {r.mean_a:>10.4f} {r.mean_b:>10.4f} {r.mean_ref:>10.4f} {r.direction:<8}"
        )
    flagged_count = len(report.flagged)
    lines.append(f"\n{flagged_count} regression(s) detected out of {len(report.results)} compared.")
    return "\n".join(lines)
