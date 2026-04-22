"""Compute a scalar 'diff score' summarising how much two checkpoints differ."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class DiffScore:
    """Aggregated numeric score for a checkpoint diff."""

    total_keys: int
    changed_keys: int
    added_keys: int
    removed_keys: int
    mean_abs_delta: float  # average |mean_a - mean_b| across changed tensors
    max_abs_delta: float   # worst-case |mean_a - mean_b|
    score: float           # normalised 0-1 summary score


def _abs_mean_delta(td: TensorDiff) -> float:
    """Return |mean_a - mean_b| for a changed TensorDiff, else 0.0."""
    if td.mean_a is None or td.mean_b is None:
        return 0.0
    return abs(float(td.mean_a) - float(td.mean_b))


def compute_score(diff: CheckpointDiff) -> DiffScore:
    """Compute a DiffScore from a CheckpointDiff."""
    changed: List[TensorDiff] = [
        td for td in diff.values() if td.status == "changed"
    ]
    added = sum(1 for td in diff.values() if td.status == "added")
    removed = sum(1 for td in diff.values() if td.status == "removed")

    deltas = [_abs_mean_delta(td) for td in changed]
    mean_abs = float(np.mean(deltas)) if deltas else 0.0
    max_abs = float(np.max(deltas)) if deltas else 0.0

    total = len(diff)
    differing = len(changed) + added + removed
    raw_ratio = differing / total if total > 0 else 0.0
    # blend key-level ratio with magnitude signal (capped at 1)
    magnitude_signal = min(mean_abs / (mean_abs + 1.0), 1.0)
    score = round(0.6 * raw_ratio + 0.4 * magnitude_signal, 4)

    return DiffScore(
        total_keys=total,
        changed_keys=len(changed),
        added_keys=added,
        removed_keys=removed,
        mean_abs_delta=round(mean_abs, 6),
        max_abs_delta=round(max_abs, 6),
        score=score,
    )


def format_score(ds: DiffScore) -> str:
    """Return a human-readable summary of a DiffScore."""
    lines = [
        f"Diff Score : {ds.score:.4f}",
        f"  Keys     : {ds.total_keys} total, "
        f"{ds.changed_keys} changed, "
        f"{ds.added_keys} added, "
        f"{ds.removed_keys} removed",
        f"  Mean |Δμ|: {ds.mean_abs_delta:.6f}",
        f"  Max  |Δμ|: {ds.max_abs_delta:.6f}",
    ]
    return "\n".join(lines)
