"""Heatmap rendering for per-key magnitude changes across a checkpoint diff."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import math

from checkpoint_diff.diff import CheckpointDiff, TensorDiff

_BLOCKS = " ▁▂▃▄▅▆▇█"
_BAR_WIDTH = 20


@dataclass
class HeatmapRow:
    key: str
    abs_mean_delta: float
    bar: str
    status: str


def _abs_mean_delta(td: TensorDiff) -> float:
    """Return |mean_b - mean_a|, or 0 if either side is absent."""
    if td.mean_a is None or td.mean_b is None:
        return 0.0
    return abs(td.mean_b - td.mean_a)


def _render_bar(value: float, max_value: float, width: int = _BAR_WIDTH) -> str:
    """Render a unicode block-character bar scaled to max_value."""
    if max_value == 0.0:
        return _BLOCKS[0] * width
    ratio = min(value / max_value, 1.0)
    filled = ratio * width
    full_blocks = int(filled)
    remainder = filled - full_blocks
    block_idx = math.floor(remainder * (len(_BLOCKS) - 1))
    bar = _BLOCKS[-1] * full_blocks
    if full_blocks < width:
        bar += _BLOCKS[block_idx]
        bar += _BLOCKS[0] * (width - full_blocks - 1)
    return bar


def build_heatmap(
    diff: CheckpointDiff,
    top_n: Optional[int] = None,
    include_unchanged: bool = False,
) -> List[HeatmapRow]:
    """Build a list of HeatmapRow objects sorted by descending abs_mean_delta."""
    rows: List[HeatmapRow] = []
    for key, td in diff.items():
        if not include_unchanged and td.status == "unchanged":
            continue
        rows.append(
            HeatmapRow(
                key=key,
                abs_mean_delta=_abs_mean_delta(td),
                bar="",  # filled below
                status=td.status,
            )
        )

    rows.sort(key=lambda r: r.abs_mean_delta, reverse=True)
    if top_n is not None:
        rows = rows[:top_n]

    max_delta = max((r.abs_mean_delta for r in rows), default=0.0)
    for row in rows:
        row.bar = _render_bar(row.abs_mean_delta, max_delta)

    return rows


def format_heatmap(rows: List[HeatmapRow]) -> str:
    """Return a formatted string table for the heatmap."""
    if not rows:
        return "(no data to display)"
    key_w = max(len(r.key) for r in rows)
    lines = [
        f"{'key':<{key_w}}  {'delta':>10}  bar",
        "-" * (key_w + 2 + 10 + 2 + _BAR_WIDTH),
    ]
    for row in rows:
        lines.append(
            f"{row.key:<{key_w}}  {row.abs_mean_delta:>10.4f}  {row.bar}  [{row.status}]"
        )
    return "\n".join(lines)
