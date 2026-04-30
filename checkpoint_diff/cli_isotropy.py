"""CLI integration for isotropy analysis."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.isotropy import compute_isotropy, format_isotropy


def add_isotropy_args(parser: argparse.ArgumentParser) -> None:
    """Register --isotropy and related flags onto *parser*."""
    parser.add_argument(
        "--isotropy",
        action="store_true",
        default=False,
        help="Show per-tensor isotropy scores (singular-value entropy method).",
    )
    parser.add_argument(
        "--isotropy-top-n",
        type=int,
        default=None,
        metavar="N",
        dest="isotropy_top_n",
        help="Limit isotropy report to the top-N tensors by absolute delta.",
    )


def apply_isotropy(
    args: argparse.Namespace,
    diff: CheckpointDiff,
    *,
    print_fn=print,
) -> Optional[str]:
    """If --isotropy was requested, compute and print the report.

    Returns the formatted string (useful for testing), or None.
    """
    if not getattr(args, "isotropy", False):
        return None
    rows = compute_isotropy(diff, top_n=getattr(args, "isotropy_top_n", None))
    report = format_isotropy(rows)
    print_fn(report)
    return report
