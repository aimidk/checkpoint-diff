"""CLI integration for quantile shift analysis."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.quantile_shift import compute_quantile_shifts, format_quantile_shifts


def add_quantile_shift_args(parser: argparse.ArgumentParser) -> None:
    """Register --quantile-shift and related flags on *parser*."""
    parser.add_argument(
        "--quantile-shift",
        action="store_true",
        default=False,
        help="Show quantile shift analysis between checkpoints.",
    )
    parser.add_argument(
        "--quantile-shift-top-n",
        type=int,
        default=None,
        metavar="N",
        help="Limit quantile shift output to the top N keys by max absolute shift.",
    )
    parser.add_argument(
        "--quantile-shift-include-unchanged",
        action="store_true",
        default=False,
        help="Include unchanged keys in quantile shift output.",
    )


def apply_quantile_shift(
    args: argparse.Namespace,
    diff: CheckpointDiff,
) -> Optional[str]:
    """Return formatted quantile shift report if the flag is set, else None."""
    if not getattr(args, "quantile_shift", False):
        return None

    rows = compute_quantile_shifts(
        diff,
        include_unchanged=getattr(args, "quantile_shift_include_unchanged", False),
        top_n=getattr(args, "quantile_shift_top_n", None),
    )
    return format_quantile_shifts(rows)
