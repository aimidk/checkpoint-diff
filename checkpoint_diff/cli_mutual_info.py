"""CLI integration for mutual information analysis."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.mutual_info import compute_mutual_info, format_mutual_info


def add_mutual_info_args(parser: argparse.ArgumentParser) -> None:
    """Register --mutual-info and related flags on *parser*."""
    parser.add_argument(
        "--mutual-info",
        action="store_true",
        default=False,
        help="Compute pairwise mutual information between tensor versions.",
    )
    parser.add_argument(
        "--mi-bins",
        type=int,
        default=64,
        metavar="N",
        help="Number of histogram bins used for MI estimation (default: 64).",
    )
    parser.add_argument(
        "--mi-top-n",
        type=int,
        default=None,
        metavar="N",
        help="Limit output to top-N keys by mutual information.",
    )


def apply_mutual_info(
    args: argparse.Namespace,
    diff: CheckpointDiff,
) -> Optional[str]:
    """Return formatted mutual-info report if the flag is set, else None."""
    if not getattr(args, "mutual_info", False):
        return None
    bins = getattr(args, "mi_bins", 64)
    top_n = getattr(args, "mi_top_n", None)
    rows = compute_mutual_info(diff, bins=bins, top_n=top_n)
    return format_mutual_info(rows)
