"""Rank tensors by various change metrics across a checkpoint diff.

Provides a unified ranking view so users can quickly identify the
most-changed, most-sparse, highest-magnitude, or highest-entropy
tensors without running each analysis separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff

RankMetric = Literal["abs_mean_delta", "abs_std_delta", "l2_norm_b", "rel_change", "sparsity_b"]

VALID_METRICS: tuple[str, ...] = (
    "abs_mean_delta",
    "abs_std_delta",
    "l2_norm_b",
    "rel_change",
    "sparsity_b",
)


@dataclass
class RankRow:
    """A single row in the ranking table."""

    key: str
    status: str
    metric: str
    value: float
    rank: int


def _abs_mean_delta(td: TensorDiff) -> float:
    """Absolute difference between mean of b and mean of a."""
    if td.mean_a is None or td.mean_b is None:
        return float("nan")
    return abs(td.mean_b - td.mean_a)


def _abs_std_delta(td: TensorDiff) -> float:
    """Absolute difference between std of b and std of a."""
    if td.std_a is None or td.std_b is None:
        return float("nan")
    return abs(td.std_b - td.std_a)


def _l2_norm_b(td: TensorDiff) -> float:
    """L2 norm of tensor b (or a if b is absent)."""
    arr = td.data_b if td.data_b is not None else td.data_a
    if arr is None:
        return float("nan")
    return float(np.linalg.norm(arr.astype(np.float64)))


def _rel_change(td: TensorDiff) -> float:
    """Relative change in L2 norm from a to b."""
    if td.data_a is None or td.data_b is None:
        return float("nan")
    norm_a = float(np.linalg.norm(td.data_a.astype(np.float64)))
    norm_b = float(np.linalg.norm(td.data_b.astype(np.float64)))
    if norm_a == 0.0:
        return float("nan")
    return abs(norm_b - norm_a) / norm_a


def _sparsity_b(td: TensorDiff) -> float:
    """Fraction of near-zero elements in tensor b (or a if b absent)."""
    arr = td.data_b if td.data_b is not None else td.data_a
    if arr is None or arr.size == 0:
        return float("nan")
    return float(np.sum(np.abs(arr) < 1e-6) / arr.size)


_METRIC_FNS: Dict[str, object] = {
    "abs_mean_delta": _abs_mean_delta,
    "abs_std_delta": _abs_std_delta,
    "l2_norm_b": _l2_norm_b,
    "rel_change": _rel_change,
    "sparsity_b": _sparsity_b,
}


def rank_tensors(
    diff: CheckpointDiff,
    metric: RankMetric = "abs_mean_delta",
    top_n: Optional[int] = None,
    ascending: bool = False,
) -> List[RankRow]:
    """Rank all tensors in *diff* by *metric*.

    Parameters
    ----------
    diff:
        The computed checkpoint diff.
    metric:
        One of the supported metric names (see ``VALID_METRICS``).
    top_n:
        If given, return only the top *n* rows.
    ascending:
        When ``True`` rank from smallest to largest value.

    Returns
    -------
    List[RankRow]
        Rows sorted by *metric*, with ``rank`` starting at 1.
    """
    if metric not in _METRIC_FNS:
        raise ValueError(f"Unknown metric {metric!r}. Choose from {VALID_METRICS}.")

    fn = _METRIC_FNS[metric]
    rows: List[RankRow] = []

    for key, td in diff.items():
        value = fn(td)  # type: ignore[operator]
        rows.append(RankRow(key=key, status=td.status, metric=metric, value=value, rank=0))

    # Sort; push NaN values to the end regardless of direction
    rows.sort(
        key=lambda r: (np.isnan(r.value), r.value if not np.isnan(r.value) else 0.0),
        reverse=not ascending,
    )

    for i, row in enumerate(rows, start=1):
        row.rank = i

    if top_n is not None:
        rows = rows[:top_n]

    return rows


def format_rank(rows: List[RankRow]) -> str:
    """Return a human-readable table of ranked tensors."""
    if not rows:
        return "No tensors to rank."

    metric_label = rows[0].metric if rows else "value"
    header = f"{'Rank':>4}  {'Key':<45}  {'Status':<10}  {metric_label:>16}"
    sep = "-" * len(header)
    lines = [header, sep]

    for row in rows:
        val_str = f"{row.value:.6f}" if not np.isnan(row.value) else "     nan"
        lines.append(f"{row.rank:>4}  {row.key:<45}  {row.status:<10}  {val_str:>16}")

    return "\n".join(lines)
