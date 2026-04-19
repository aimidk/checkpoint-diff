"""Format and export trend history reports."""
from __future__ import annotations

from typing import List
import csv
import io

from checkpoint_diff.history import KeyTrend


def format_trend_table(trends: List[KeyTrend], top_n: int = 10) -> str:
    """Return a human-readable table of key trends sorted by total delta."""
    sorted_trends = sorted(trends, key=lambda t: abs(t.total_delta), reverse=True)
    top = sorted_trends[:top_n]

    if not top:
        return "No trends to display.\n"

    header = f"{'Key':<40} {'Steps':>6} {'Total Δ':>12} {'Max Δ':>12} {'Direction'}"
    sep = "-" * len(header)
    lines = [header, sep]

    for t in top:
        direction = "↑" if t.total_delta > 0 else ("↓" if t.total_delta < 0 else "─")
        lines.append(
            f"{t.key:<40} {t.num_steps:>6} {t.total_delta:>12.6f} {t.max_delta:>12.6f} {direction}"
        )

    return "\n".join(lines) + "\n"


def export_trends_csv(trends: List[KeyTrend]) -> str:
    """Serialize trends to CSV string."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["key", "num_steps", "total_delta", "max_delta", "deltas"])
    for t in trends:
        writer.writerow([
            t.key,
            t.num_steps,
            t.total_delta,
            t.max_delta,
            "|".join(f"{d:.6f}" for d in t.deltas),
        ])
    return buf.getvalue()


def print_trend_report(trends: List[KeyTrend], top_n: int = 10) -> None:
    """Print trend table to stdout."""
    print(format_trend_table(trends, top_n=top_n))
