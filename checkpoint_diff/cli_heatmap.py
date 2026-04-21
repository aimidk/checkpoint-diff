"""CLI integration for the heatmap feature."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.heatmap import build_heatmap, format_heatmap


def add_heatmap_args(parser: argparse.ArgumentParser) -> None:
    """Register heatmap-related flags on *parser*."""
    grp = parser.add_argument_group("heatmap")
    grp.add_argument(
        "--heatmap",
        action="store_true",
        default=False,
        help="Print a magnitude-change heatmap for all keys.",
    )
    grp.add_argument(
        "--heatmap-top",
        type=int,
        default=None,
        metavar="N",
        dest="heatmap_top",
        help="Limit heatmap to the top N keys by abs mean delta.",
    )
    grp.add_argument(
        "--heatmap-unchanged",
        action="store_true",
        default=False,
        dest="heatmap_unchanged",
        help="Include unchanged keys in the heatmap.",
    )


def apply_heatmap(
    args: argparse.Namespace,
    diff: CheckpointDiff,
) -> Optional[str]:
    """If --heatmap is set, build and return the formatted heatmap string.

    Returns *None* when the flag is not set so callers can skip output.
    """
    if not getattr(args, "heatmap", False):
        return None

    rows = build_heatmap(
        diff,
        top_n=getattr(args, "heatmap_top", None),
        include_unchanged=getattr(args, "heatmap_unchanged", False),
    )
    return format_heatmap(rows)
