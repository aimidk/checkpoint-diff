"""CLI integration for the variance / coefficient-of-variation feature."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.variance import compute_variance, format_variance


def add_variance_args(parser: argparse.ArgumentParser) -> None:
    """Attach variance-related flags to *parser*."""
    grp = parser.add_argument_group("variance")
    grp.add_argument(
        "--variance",
        action="store_true",
        default=False,
        help="Show per-key variance and coefficient-of-variation analysis.",
    )
    grp.add_argument(
        "--variance-top-n",
        type=int,
        default=None,
        metavar="N",
        dest="variance_top_n",
        help="Limit variance table to the top N keys by absolute variance delta.",
    )
    grp.add_argument(
        "--variance-include-unchanged",
        action="store_true",
        default=False,
        dest="variance_include_unchanged",
        help="Include unchanged keys in the variance table.",
    )


def apply_variance(
    args: argparse.Namespace,
    diff: CheckpointDiff,
) -> Optional[str]:
    """If the --variance flag is set, compute and return the formatted table.

    Returns *None* when the flag is not set so callers can skip output.
    """
    if not getattr(args, "variance", False):
        return None
    rows = compute_variance(
        diff,
        include_unchanged=getattr(args, "variance_include_unchanged", False),
    )
    return format_variance(rows, top_n=getattr(args, "variance_top_n", None))
