"""Drift detection: flag keys whose statistics have drifted beyond a rolling baseline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class DriftResult:
    key: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    mean_drift: float
    std_drift: float
    flagged: bool


@dataclass
class DriftReport:
    results: List[DriftResult] = field(default_factory=list)

    @property
    def flagged(self) -> List[DriftResult]:
        return [r for r in self.results if r.flagged]


def _safe_rel(new: float, old: float) -> float:
    """Relative change from old to new; falls back to absolute when old is ~0."""
    if abs(old) < 1e-12:
        return abs(new - old)
    return abs((new - old) / old)


def detect_drift(
    diff: CheckpointDiff,
    mean_threshold: float = 0.1,
    std_threshold: float = 0.1,
    include_unchanged: bool = False,
) -> DriftReport:
    """Detect statistical drift for tensors present in both checkpoints.

    Args:
        diff: A computed :class:`CheckpointDiff`.
        mean_threshold: Relative change in mean that triggers a flag.
        std_threshold: Relative change in std that triggers a flag.
        include_unchanged: If True, include unchanged tensors in the report.

    Returns:
        A :class:`DriftReport` containing per-key results.
    """
    results: List[DriftResult] = []

    for key, td in diff.items():
        if td.status in ("added", "removed"):
            continue
        if td.stats_a is None or td.stats_b is None:
            continue
        mean_drift = _safe_rel(td.stats_b.mean, td.stats_a.mean)
        std_drift = _safe_rel(td.stats_b.std, td.stats_a.std)
        flagged = mean_drift > mean_threshold or std_drift > std_threshold
        if not flagged and not include_unchanged:
            continue
        results.append(
            DriftResult(
                key=key,
                mean_a=td.stats_a.mean,
                mean_b=td.stats_b.mean,
                std_a=td.stats_a.std,
                std_b=td.stats_b.std,
                mean_drift=mean_drift,
                std_drift=std_drift,
                flagged=flagged,
            )
        )

    results.sort(key=lambda r: max(r.mean_drift, r.std_drift), reverse=True)
    return DriftReport(results=results)


def format_drift(report: DriftReport, top_n: Optional[int] = None) -> str:
    """Render a drift report as a human-readable table."""
    rows = report.results[:top_n] if top_n else report.results
    if not rows:
        return "No drift detected."
    header = f"{'Key':<40} {'MeanΔ%':>8} {'StdΔ%':>8} {'Flag':>6}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in rows:
        flag = "!" if r.flagged else " "
        lines.append(
            f"{r.key:<40} {r.mean_drift * 100:>7.2f}% {r.std_drift * 100:>7.2f}% {flag:>6}"
        )
    return "\n".join(lines)
