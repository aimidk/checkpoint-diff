"""CLI integration for effective rank analysis."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.effective_rank import compute_effective_rank, format_effective_rank


def add_effective_rank_args(parser: argparse.ArgumentParser) -> None:
    """Register --effective-rank and related flags on *parser*."""
    parser.add_argument(
        "--effective-rank",
        action="store_true",
        default=False,
        help="Show effective rank (SVD entropy) for each tensor.",
    )
    parser.add_argument(
        "--effective-rank-top-n",
        type=int,
        default=None,
        metavar="N",
        help="Limit effective rank output to top N rows by absolute delta.",
    )


def apply_effective_rank(
    args: argparse.Namespace,
    diff: CheckpointDiff,
) -> Optional[str]:
    """Return formatted effective rank table if the flag is set, else None."""
    if not getattr(args, "effective_rank", False):
        return None
    rows = compute_effective_rank(diff)
    top_n: Optional[int] = getattr(args, "effective_rank_top_n", None)
    return format_effective_rank(rows, top_n=top_n)
