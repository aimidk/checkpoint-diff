"""CLI integration for skewness analysis."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.skewness import compute_skewness, format_skewness


def add_skewness_args(parser: argparse.ArgumentParser) -> None:
    """Register --skewness and related flags on *parser*."""
    parser.add_argument(
        "--skewness",
        action="store_true",
        default=False,
        help="Show per-tensor skewness comparison.",
    )
    parser.add_argument(
        "--skewness-top-n",
        type=int,
        default=None,
        metavar="N",
        dest="skewness_top_n",
        help="Limit skewness output to the top N tensors by absolute delta.",
    )


def apply_skewness(
    args: argparse.Namespace,
    diff: CheckpointDiff,
    top_n: Optional[int] = None,
) -> Optional[str]:
    """If --skewness is set, compute and return the formatted report."""
    if not getattr(args, "skewness", False):
        return None
    n = getattr(args, "skewness_top_n", None) if top_n is None else top_n
    rows = compute_skewness(diff)
    return format_skewness(rows, top_n=n)
