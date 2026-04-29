"""CLI integration for kurtosis analysis."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.kurtosis import compute_kurtosis, format_kurtosis


def add_kurtosis_args(parser: argparse.ArgumentParser) -> None:
    """Register kurtosis-related flags on *parser*."""
    grp = parser.add_argument_group("kurtosis")
    grp.add_argument(
        "--kurtosis",
        action="store_true",
        default=False,
        help="Show excess kurtosis for each tensor weight distribution.",
    )
    grp.add_argument(
        "--kurtosis-include-unchanged",
        action="store_true",
        default=False,
        help="Include unchanged tensors in kurtosis output.",
    )
    grp.add_argument(
        "--kurtosis-top-n",
        type=int,
        default=None,
        metavar="N",
        help="Limit kurtosis output to the top N tensors by absolute delta.",
    )


def apply_kurtosis(
    args: argparse.Namespace,
    diff: CheckpointDiff,
) -> Optional[str]:
    """Return formatted kurtosis report if the flag is set, else None."""
    if not getattr(args, "kurtosis", False):
        return None
    rows = compute_kurtosis(
        diff,
        include_unchanged=getattr(args, "kurtosis_include_unchanged", False),
        top_n=getattr(args, "kurtosis_top_n", None),
    )
    return format_kurtosis(rows)
