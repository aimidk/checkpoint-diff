"""CLI integration for mutual information analysis."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.mutual_info import compute_mutual_info, format_mutual_info


def add_mutual_info_args(parser: argparse.ArgumentParser) -> None:
    """Register --mutual-info flags on *parser*."""
    grp = parser.add_argument_group("mutual information")
    grp.add_argument(
        "--mutual-info",
        action="store_true",
        default=False,
        help="Show mutual information between tensor pairs.",
    )
    grp.add_argument(
        "--mi-bins",
        type=int,
        default=32,
        metavar="N",
        help="Number of histogram bins for MI estimation (default: 32).",
    )
    grp.add_argument(
        "--mi-top-n",
        type=int,
        default=None,
        metavar="N",
        help="Limit output to top N keys by mutual information.",
    )


def apply_mutual_info(
    args: argparse.Namespace,
    diff: CheckpointDiff,
) -> Optional[str]:
    """Return formatted mutual-info report when --mutual-info is set, else None."""
    if not getattr(args, "mutual_info", False):
        return None
    rows = compute_mutual_info(
        diff,
        bins=getattr(args, "mi_bins", 32),
        top_n=getattr(args, "mi_top_n", None),
    )
    return format_mutual_info(rows)
