"""CLI integration for sparsity analysis."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.sparsity import compute_sparsity, format_sparsity


def add_sparsity_args(parser: argparse.ArgumentParser) -> None:
    """Register sparsity-related flags on *parser*."""
    grp = parser.add_argument_group("sparsity")
    grp.add_argument(
        "--sparsity",
        action="store_true",
        default=False,
        help="Show zero-fraction and near-zero statistics for each tensor.",
    )
    grp.add_argument(
        "--sparsity-eps",
        type=float,
        default=1e-6,
        metavar="EPS",
        help="Threshold below which a value is considered near-zero (default: 1e-6).",
    )
    grp.add_argument(
        "--sparsity-top-n",
        type=int,
        default=None,
        metavar="N",
        help="Limit sparsity report to the top N rows by |delta_sparsity|.",
    )
    grp.add_argument(
        "--sparsity-include-unchanged",
        action="store_true",
        default=False,
        help="Include unchanged tensors in the sparsity report.",
    )


def apply_sparsity(
    args: argparse.Namespace,
    diff: CheckpointDiff,
) -> Optional[str]:
    """Return a formatted sparsity report if ``--sparsity`` was requested."""
    if not getattr(args, "sparsity", False):
        return None
    rows = compute_sparsity(
        diff,
        eps=args.sparsity_eps,
        include_unchanged=args.sparsity_include_unchanged,
    )
    return format_sparsity(rows, top_n=args.sparsity_top_n)
