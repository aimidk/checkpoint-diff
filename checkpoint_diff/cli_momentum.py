"""CLI integration for momentum analysis."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.momentum import compute_momentum, format_momentum


def add_momentum_args(parser: argparse.ArgumentParser) -> None:
    """Register --momentum and related flags on *parser*."""
    parser.add_argument(
        "--momentum",
        action="store_true",
        default=False,
        help="Show effective update magnitude (momentum proxy) per key.",
    )
    parser.add_argument(
        "--momentum-top-n",
        type=int,
        default=None,
        metavar="N",
        dest="momentum_top_n",
        help="Limit momentum report to top N keys by relative momentum.",
    )
    parser.add_argument(
        "--momentum-export",
        type=str,
        default=None,
        metavar="FILE",
        dest="momentum_export",
        help="Write momentum report as CSV to FILE.",
    )


def apply_momentum(
    args: argparse.Namespace,
    diff: CheckpointDiff,
) -> Optional[str]:
    """Compute and return the momentum report if --momentum was requested.

    Side-effect: writes CSV file when --momentum-export is set.
    Returns formatted string or None.
    """
    if not getattr(args, "momentum", False):
        return None

    rows = compute_momentum(diff)
    top_n: Optional[int] = getattr(args, "momentum_top_n", None)
    report = format_momentum(rows, top_n=top_n)

    export_path: Optional[str] = getattr(args, "momentum_export", None)
    if export_path:
        import csv
        with open(export_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["key", "delta_l2", "base_l2", "rel_momentum", "status"])
            for r in rows:
                writer.writerow([r.key, r.delta_l2, r.base_l2, r.rel_momentum, r.status])

    return report
