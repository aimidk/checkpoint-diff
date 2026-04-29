"""CLI integration for per-layer norm analysis."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.layer_norm import compute_layer_norms, format_layer_norms


def add_layer_norm_args(parser: argparse.ArgumentParser) -> None:
    """Register --layer-norm and related flags on *parser*."""
    group = parser.add_argument_group("layer norm")
    group.add_argument(
        "--layer-norm",
        action="store_true",
        default=False,
        help="Show per-layer L1/L2 norm analysis.",
    )
    group.add_argument(
        "--layer-norm-top-n",
        type=int,
        default=None,
        metavar="N",
        help="Limit layer-norm output to the top N rows by |ΔL2|.",
    )
    group.add_argument(
        "--layer-norm-export",
        default=None,
        metavar="FILE",
        help="Export layer-norm table to a CSV file.",
    )


def apply_layer_norm(
    args: argparse.Namespace,
    diff: CheckpointDiff,
) -> Optional[str]:
    """Compute and return the formatted layer-norm report if requested.

    Also writes a CSV export when ``--layer-norm-export`` is provided.
    Returns *None* when the feature is disabled.
    """
    if not getattr(args, "layer_norm", False):
        return None

    top_n: Optional[int] = getattr(args, "layer_norm_top_n", None)
    export_path: Optional[str] = getattr(args, "layer_norm_export", None)

    rows = compute_layer_norms(diff)

    if export_path:
        import csv
        with open(export_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["key", "status", "l1_a", "l1_b", "l2_a", "l2_b", "l1_delta", "l2_delta"])
            for r in rows:
                writer.writerow([r.key, r.status, r.l1_a, r.l1_b, r.l2_a, r.l2_b, r.l1_delta, r.l2_delta])

    return format_layer_norms(rows, top_n=top_n)
