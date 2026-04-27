"""CLI integration for magnitude analysis."""
from __future__ import annotations

import argparse
from typing import Optional

from .diff import CheckpointDiff
from .magnitude import compute_magnitude, format_magnitude


def add_magnitude_args(parser: argparse.ArgumentParser) -> None:
    """Register --magnitude and related flags on *parser*."""
    parser.add_argument(
        "--magnitude",
        action="store_true",
        default=False,
        help="Show per-tensor L1/L2 norm magnitude report.",
    )
    parser.add_argument(
        "--magnitude-top-n",
        type=int,
        default=None,
        metavar="N",
        help="Limit magnitude report to top N tensors by relative L2 change.",
    )
    parser.add_argument(
        "--magnitude-export",
        default=None,
        metavar="FILE",
        help="Export magnitude report to a CSV file.",
    )


def apply_magnitude(args: argparse.Namespace, diff: CheckpointDiff) -> Optional[str]:
    """Compute and return a formatted magnitude report if requested.

    Side-effect: writes CSV when *--magnitude-export* is set.
    Returns the formatted string or None.
    """
    if not getattr(args, "magnitude", False):
        return None

    top_n: Optional[int] = getattr(args, "magnitude_top_n", None)
    rows = compute_magnitude(diff)
    report = format_magnitude(rows, top_n=top_n)

    export_path: Optional[str] = getattr(args, "magnitude_export", None)
    if export_path:
        import csv
        with open(export_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["key", "status", "norm_l1_a", "norm_l1_b",
                             "norm_l2_a", "norm_l2_b", "rel_change_l2"])
            for r in rows:
                writer.writerow([r.key, r.status, r.norm_l1_a, r.norm_l1_b,
                                 r.norm_l2_a, r.norm_l2_b,
                                 "" if r.rel_change_l2 is None else r.rel_change_l2])
    return report
