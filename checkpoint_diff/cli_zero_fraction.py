"""CLI integration for zero-fraction analysis."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.zero_fraction import compute_zero_fractions, format_zero_fractions


def add_zero_fraction_args(parser: argparse.ArgumentParser) -> None:
    """Register --zero-fraction and --zero-fraction-top-n flags on *parser*."""
    parser.add_argument(
        "--zero-fraction",
        action="store_true",
        default=False,
        help="Show fraction of zero-valued elements for each changed tensor.",
    )
    parser.add_argument(
        "--zero-fraction-top-n",
        type=int,
        default=None,
        metavar="N",
        help="Limit zero-fraction output to the top N rows by absolute delta.",
    )


def apply_zero_fraction(
    args: argparse.Namespace,
    diff: CheckpointDiff,
    *,
    print_fn=print,
) -> None:
    """If --zero-fraction is set, compute and print the zero-fraction table."""
    if not getattr(args, "zero_fraction", False):
        return
    top_n: Optional[int] = getattr(args, "zero_fraction_top_n", None)
    rows = compute_zero_fractions(diff)
    print_fn(format_zero_fractions(rows, top_n=top_n))
