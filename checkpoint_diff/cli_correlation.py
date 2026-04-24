"""CLI integration for the correlation feature."""
from __future__ import annotations

import argparse
from typing import List

from checkpoint_diff.correlation import CorrelationRow, compute_correlations, format_correlations
from checkpoint_diff.diff import CheckpointDiff


def add_correlation_args(parser: argparse.ArgumentParser) -> None:
    """Register --correlation and related flags on *parser*."""
    parser.add_argument(
        "--correlation",
        action="store_true",
        default=False,
        help="Show Pearson correlation between weight tensors.",
    )
    parser.add_argument(
        "--correlation-include-unchanged",
        action="store_true",
        default=False,
        dest="correlation_include_unchanged",
        help="Include unchanged tensors in the correlation table.",
    )
    parser.add_argument(
        "--correlation-top-n",
        type=int,
        default=None,
        dest="correlation_top_n",
        metavar="N",
        help="Limit output to the N most negatively correlated tensors.",
    )


def apply_correlation(
    args: argparse.Namespace,
    diff: CheckpointDiff,
) -> List[CorrelationRow]:
    """Compute and print correlations when --correlation is set."""
    if not getattr(args, "correlation", False):
        return []
    rows = compute_correlations(
        diff,
        include_unchanged=args.correlation_include_unchanged,
    )
    print(format_correlations(rows, top_n=args.correlation_top_n))
    return rows
