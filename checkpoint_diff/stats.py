"""Per-tensor descriptive statistics beyond what TensorDiff already stores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class TensorStats:
    key: str
    min: float
    max: float
    mean: float
    std: float
    median: float
    l2_norm: float
    num_elements: int


def _compute_stats(key: str, td: TensorDiff) -> TensorStats:
    """Compute descriptive statistics for the *new* (or only) tensor in a diff."""
    arr = td.tensor_b if td.tensor_b is not None else td.tensor_a
    if arr is None:
        raise ValueError(f"No tensor data available for key '{key}'")
    flat = arr.astype(np.float64).ravel()
    return TensorStats(
        key=key,
        min=float(np.min(flat)),
        max=float(np.max(flat)),
        mean=float(np.mean(flat)),
        std=float(np.std(flat)),
        median=float(np.median(flat)),
        l2_norm=float(np.linalg.norm(flat)),
        num_elements=int(flat.size),
    )


def compute_stats(diff: CheckpointDiff) -> Dict[str, TensorStats]:
    """Return a mapping of key -> TensorStats for all non-removed tensors."""
    result: Dict[str, TensorStats] = {}
    for key, td in diff.items():
        if td.tensor_b is None and td.tensor_a is not None:
            # removed key — skip (no 'new' tensor)
            continue
        try:
            result[key] = _compute_stats(key, td)
        except ValueError:
            pass
    return result


def format_stats(stats: Dict[str, TensorStats], top_n: int = 0) -> str:
    """Format a human-readable stats table.

    Args:
        stats: mapping returned by :func:`compute_stats`.
        top_n: if > 0, limit output to the *top_n* keys with the highest L2 norm.
    """
    if not stats:
        return "No statistics available."

    rows: List[TensorStats] = sorted(
        stats.values(), key=lambda s: s.l2_norm, reverse=True
    )
    if top_n > 0:
        rows = rows[:top_n]

    header = f"{'Key':<40} {'min':>10} {'max':>10} {'mean':>10} {'std':>10} {'l2':>12} {'numel':>10}"
    sep = "-" * len(header)
    lines = [header, sep]
    for s in rows:
        lines.append(
            f"{s.key:<40} {s.min:>10.4f} {s.max:>10.4f} "
            f"{s.mean:>10.4f} {s.std:>10.4f} {s.l2_norm:>12.4f} {s.num_elements:>10}"
        )
    return "\n".join(lines)
