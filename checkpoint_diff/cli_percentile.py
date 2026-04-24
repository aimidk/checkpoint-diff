"""CLI integration for percentile analysis."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.percentile import compute_percentiles, format_percentiles


def add_percentile_args(parser: argparse.ArgumentParser) -> None:
    """Register percentile-related flags on *parser*."""
    grp = parser.add_argument_group("percentile analysis")
    grp.add_argument(
        "--percentiles",
        action="store_true",
        default=False,
        help="Show per-percentile weight distribution for changed tensors.",
    )
    grp.add_argument(
        "--percentile-keys",
        nargs="+",
        metavar="KEY",
        default=None,
        dest="percentile_keys",
        help="Restrict percentile analysis to these tensor keys.",
    )
    grp.add_argument(
        "--percentile-abs",
        action="store_true",
        default=False,
        dest="percentile_abs",
        help="Show absolute values for checkpoint B instead of deltas.",
    )


def apply_percentile(
    args: argparse.Namespace,
    diff: CheckpointDiff,
    *,
    output_lines: Optional[list] = None,
) -> Optional[str]:
    """Run percentile analysis if requested; return formatted string or None."""
    if not getattr(args, "percentiles", False):
        return None

    keys = getattr(args, "percentile_keys", None)
    show_delta = not getattr(args, "percentile_abs", False)
    rows = compute_percentiles(diff, keys=keys)
    report = format_percentiles(rows, show_delta=show_delta)

    if output_lines is not None:
        output_lines.append(report)

    return report
