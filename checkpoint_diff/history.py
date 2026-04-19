"""history.py – compare a sequence of checkpoints and track metric trends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .diff import CheckpointDiff, compute_diff
from .loader import load_checkpoint


@dataclass
class StepDiff:
    """Diff between two consecutive checkpoints in a history."""

    step_a: int
    step_b: int
    diff: CheckpointDiff


@dataclass
class KeyTrend:
    """Mean-absolute-value trend for a single tensor key across steps."""

    key: str
    steps: List[int] = field(default_factory=list)
    mean_abs: List[float] = field(default_factory=list)

    @property
    def delta(self) -> Optional[float]:
        """Total change from first to last recorded step."""
        if len(self.mean_abs) < 2:
            return None
        return self.mean_abs[-1] - self.mean_abs[0]


def load_history(
    paths: Sequence[str],
    steps: Optional[Sequence[int]] = None,
) -> List[StepDiff]:
    """Load a sequence of checkpoint files and compute pairwise diffs.

    Args:
        paths: Ordered list of checkpoint file paths.
        steps: Optional step indices corresponding to each path.  Defaults to
               ``0, 1, 2, …``.

    Returns:
        A list of :class:`StepDiff` objects for each consecutive pair.
    """
    if len(paths) < 2:
        raise ValueError("At least two checkpoints are required to build a history.")

    indices = list(steps) if steps is not None else list(range(len(paths)))
    if len(indices) != len(paths):
        raise ValueError("'steps' length must match 'paths' length.")

    checkpoints = [load_checkpoint(p) for p in paths]

    history: List[StepDiff] = []
    for i in range(len(checkpoints) - 1):
        diff = compute_diff(checkpoints[i], checkpoints[i + 1])
        history.append(StepDiff(step_a=indices[i], step_b=indices[i + 1], diff=diff))

    return history


def compute_trends(history: List[StepDiff]) -> Dict[str, KeyTrend]:
    """Aggregate per-key mean-absolute-value trends across a history.

    Args:
        history: Output of :func:`load_history`.

    Returns:
        Mapping of tensor key → :class:`KeyTrend`.
    """
    trends: Dict[str, KeyTrend] = {}

    for step_diff in history:
        for key, td in step_diff.diff.items():
            if td.tensor_a is None:
                continue  # skip added tensors – no baseline value
            arr = td.tensor_a
            mean_abs = float(np.mean(np.abs(arr)))
            if key not in trends:
                trends[key] = KeyTrend(key=key)
            if not trends[key].steps or trends[key].steps[-1] != step_diff.step_a:
                trends[key].steps.append(step_diff.step_a)
                trends[key].mean_abs.append(mean_abs)

        # Capture the final step values from tensor_b
        last_step = history[-1]
        if step_diff is last_step:
            for key, td in step_diff.diff.items():
                if td.tensor_b is None:
                    continue
                arr = td.tensor_b
                mean_abs = float(np.mean(np.abs(arr)))
                if key not in trends:
                    trends[key] = KeyTrend(key=key)
                trends[key].steps.append(step_diff.step_b)
                trends[key].mean_abs.append(mean_abs)

    return trends


def format_trends(trends: Dict[str, KeyTrend], top_n: int = 10) -> str:
    """Return a human-readable summary of the most-changing keys."""
    ranked = sorted(
        [t for t in trends.values() if t.delta is not None],
        key=lambda t: abs(t.delta),  # type: ignore[arg-type]
        reverse=True,
    )[:top_n]

    if not ranked:
        return "No trend data available.\n"

    lines = [f"{'Key':<40} {'Start':>10} {'End':>10} {'Delta':>10}"]
    lines.append("-" * 74)
    for t in ranked:
        lines.append(
            f"{t.key:<40} {t.mean_abs[0]:>10.4f} {t.mean_abs[-1]:>10.4f} {t.delta:>+10.4f}"
        )
    return "\n".join(lines) + "\n"
