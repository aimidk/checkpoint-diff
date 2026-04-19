"""Threshold-based filtering and flagging of tensor diffs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class ThresholdConfig:
    max_mean_delta: Optional[float] = None
    max_std_delta: Optional[float] = None
    max_max_delta: Optional[float] = None


@dataclass
class FlaggedTensor:
    key: str
    reason: str
    value: float
    threshold: float


def _delta(a, b) -> float:
    return abs(float(a) - float(b))


def flag_tensors(
    diff: CheckpointDiff,
    config: ThresholdConfig,
) -> List[FlaggedTensor]:
    """Return list of tensors that exceed any configured threshold."""
    flagged: List[FlaggedTensor] = []

    for key, td in diff.changed.items():
        if config.max_mean_delta is not None:
            delta = _delta(td.mean_a, td.mean_b)
            if delta > config.max_mean_delta:
                flagged.append(FlaggedTensor(key, "mean_delta", delta, config.max_mean_delta))

        if config.max_std_delta is not None:
            delta = _delta(td.std_a, td.std_b)
            if delta > config.max_std_delta:
                flagged.append(FlaggedTensor(key, "std_delta", delta, config.max_std_delta))

        if config.max_max_delta is not None:
            delta = _delta(td.max_a, td.max_b)
            if delta > config.max_max_delta:
                flagged.append(FlaggedTensor(key, "max_delta", delta, config.max_max_delta))

    return flagged


def format_flagged(flagged: List[FlaggedTensor]) -> str:
    if not flagged:
        return "No tensors exceeded thresholds."
    lines = ["Flagged tensors:"]
    for f in flagged:
        lines.append(
            f"  {f.key}: {f.reason}={f.value:.6g} (threshold={f.threshold:.6g})"
        )
    return "\n".join(lines)
