"""CLI integration for weight norm analysis."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.weight_norm import compute_weight_norms, format_weight_norms


def add_weight_norm_args(parser: argparse.ArgumentParser) -> None:
    """Register --weight-norm and related flags on *parser*."""
    parser.add_argument(
        "--weight-norm",
        action="store_true",
        default=False,
        help="Show per-layer L∞ and Frobenius norm comparison.",
    )
    parser.add_argument(
        "--weight-norm-top-n",
        type=int,
        default=None,
        metavar="N",
        help="Limit weight norm output to top N rows by |ΔFrob|.",
    )


def apply_weight_norm(
    args: argparse.Namespace,
    diff: CheckpointDiff,
    *,
    print_fn=print,
) -> Optional[str]:
    """If --weight-norm is set, compute and print the report; return the text."""
    if not getattr(args, "weight_norm", False):
        return None
    top_n: Optional[int] = getattr(args, "weight_norm_top_n", None)
    rows = compute_weight_norms(diff)
    text = format_weight_norms(rows, top_n=top_n)
    print_fn(text)
    return text
