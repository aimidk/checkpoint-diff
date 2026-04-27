"""CLI integration for gradient norm analysis."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.gradient import compute_gradient_norms, format_gradient_norms


def add_gradient_args(parser: argparse.ArgumentParser) -> None:
    """Register --gradient and related flags on *parser*."""
    parser.add_argument(
        "--gradient",
        action="store_true",
        default=False,
        help="Show L2 gradient norm analysis for each tensor.",
    )
    parser.add_argument(
        "--gradient-top-n",
        type=int,
        default=None,
        metavar="N",
        help="Limit gradient norm output to top N keys by absolute norm delta.",
    )
    parser.add_argument(
        "--gradient-threshold",
        type=float,
        default=None,
        metavar="T",
        help="Only show keys whose absolute norm delta exceeds T.",
    )


def apply_gradient(
    args: argparse.Namespace,
    diff: CheckpointDiff,
) -> Optional[str]:
    """Return formatted gradient norm table if --gradient is set, else None."""
    if not getattr(args, "gradient", False):
        return None
    top_n: Optional[int] = getattr(args, "gradient_top_n", None)
    threshold: Optional[float] = getattr(args, "gradient_threshold", None)
    rows = compute_gradient_norms(diff, top_n=top_n)
    if threshold is not None:
        import math
        rows = [r for r in rows if not math.isnan(r.norm_delta) and abs(r.norm_delta) >= threshold]
    return format_gradient_norms(rows)
