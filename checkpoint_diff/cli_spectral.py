"""CLI integration for spectral energy analysis."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.spectral import compute_spectral, format_spectral


def add_spectral_args(parser: argparse.ArgumentParser) -> None:
    """Register --spectral, --spectral-top-k, and --spectral-include-unchanged flags."""
    parser.add_argument(
        "--spectral",
        action="store_true",
        default=False,
        help="Show spectral energy (top-k singular value fraction) for each tensor.",
    )
    parser.add_argument(
        "--spectral-top-k",
        type=int,
        default=5,
        metavar="K",
        help="Number of top singular values to use for energy calculation (default: 5).",
    )
    parser.add_argument(
        "--spectral-include-unchanged",
        action="store_true",
        default=False,
        help="Include unchanged tensors in spectral output.",
    )
    parser.add_argument(
        "--spectral-top-n",
        type=int,
        default=None,
        metavar="N",
        help="Limit spectral output to the top N tensors by absolute delta.",
    )


def apply_spectral(args: argparse.Namespace, diff: CheckpointDiff) -> Optional[str]:
    """Compute and return formatted spectral report if --spectral is set."""
    if not getattr(args, "spectral", False):
        return None
    top_k: int = getattr(args, "spectral_top_k", 5)
    include_unchanged: bool = getattr(args, "spectral_include_unchanged", False)
    top_n: Optional[int] = getattr(args, "spectral_top_n", None)
    rows = compute_spectral(diff, top_k=top_k, include_unchanged=include_unchanged)
    if top_n is not None:
        rows = rows[:top_n]
    return format_spectral(rows)
