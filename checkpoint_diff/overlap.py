"""Compute key overlap statistics between two checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class OverlapResult:
    total_keys: int
    keys_in_both: int
    keys_only_in_a: int
    keys_only_in_b: int
    jaccard: float
    dice: float


def _jaccard(intersection: int, union: int) -> float:
    if union == 0:
        return float("nan")
    return intersection / union


def _dice(intersection: int, total_a: int, total_b: int) -> float:
    denom = total_a + total_b
    if denom == 0:
        return float("nan")
    return 2 * intersection / denom


def compute_overlap(diff: CheckpointDiff) -> OverlapResult:
    """Compute key-level overlap statistics from a CheckpointDiff."""
    keys_only_a: list[str] = []
    keys_only_b: list[str] = []
    keys_both: list[str] = []

    for key, td in diff.items():
        if td.status == "removed":
            keys_only_a.append(key)
        elif td.status == "added":
            keys_only_b.append(key)
        else:
            keys_both.append(key)

    n_both = len(keys_both)
    n_a = len(keys_only_a)
    n_b = len(keys_only_b)
    total_keys = n_both + n_a + n_b
    union = n_both + n_a + n_b

    total_a = n_both + n_a
    total_b = n_both + n_b

    return OverlapResult(
        total_keys=total_keys,
        keys_in_both=n_both,
        keys_only_in_a=n_a,
        keys_only_in_b=n_b,
        jaccard=_jaccard(n_both, union),
        dice=_dice(n_both, total_a, total_b),
    )


def format_overlap(result: OverlapResult) -> str:
    """Return a human-readable summary of the overlap result."""
    lines = [
        "Key Overlap",
        "-" * 30,
        f"  Total keys      : {result.total_keys}",
        f"  In both         : {result.keys_in_both}",
        f"  Only in A       : {result.keys_only_in_a}",
        f"  Only in B       : {result.keys_only_in_b}",
        f"  Jaccard index   : {result.jaccard:.4f}",
        f"  Dice coefficient: {result.dice:.4f}",
    ]
    return "\n".join(lines)
