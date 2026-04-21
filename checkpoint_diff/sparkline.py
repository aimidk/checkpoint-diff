"""Sparkline renderer for visualizing per-key metric trends in the terminal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

_BLOCKS = " ▁▂▃▄▅▆▇█"


@dataclass
class SparklineRow:
    key: str
    values: List[float]
    sparkline: str
    min_val: float
    max_val: float


def _normalize(values: Sequence[float]) -> List[float]:
    """Normalize values to [0, 1]. Returns zeros if range is zero."""
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def render_sparkline(values: Sequence[float]) -> str:
    """Render a sequence of floats as a Unicode sparkline string."""
    if not values:
        return ""
    normed = _normalize(values)
    n = len(_BLOCKS) - 1
    return "".join(_BLOCKS[round(v * n)] for v in normed)


def build_sparklines(
    trends: dict[str, List[float]],
    top_n: int | None = None,
) -> List[SparklineRow]:
    """Build SparklineRow entries from a mapping of key -> list of metric values.

    Args:
        trends: Mapping of tensor key to ordered list of scalar values (e.g. abs-mean).
        top_n: If given, limit output to the top_n keys with the largest range.

    Returns:
        List of SparklineRow sorted by descending value range.
    """
    rows: List[SparklineRow] = []
    for key, values in trends.items():
        if not values:
            continue
        lo = min(values)
        hi = max(values)
        spark = render_sparkline(values)
        rows.append(SparklineRow(key=key, values=list(values), sparkline=spark, min_val=lo, max_val=hi))

    rows.sort(key=lambda r: r.max_val - r.min_val, reverse=True)
    if top_n is not None:
        rows = rows[:top_n]
    return rows


def format_sparklines(rows: List[SparklineRow], width: int = 40) -> str:
    """Format SparklineRow list into a human-readable table string."""
    if not rows:
        return "(no sparkline data)"
    key_w = max(len(r.key) for r in rows)
    lines = [f"{'Key':<{key_w}}  {'Trend':<{width}}  Min        Max"]
    lines.append("-" * (key_w + width + 26))
    for r in rows:
        lines.append(
            f"{r.key:<{key_w}}  {r.sparkline:<{width}}  {r.min_val:>10.4f} {r.max_val:>10.4f}"
        )
    return "\n".join(lines)
