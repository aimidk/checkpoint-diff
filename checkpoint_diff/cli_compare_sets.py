"""CLI helpers for the compare-sets feature."""
from __future__ import annotations

import argparse
from typing import List, Optional

from checkpoint_diff.compare_sets import format_compare_set


def add_compare_sets_args(parser: argparse.ArgumentParser) -> None:
    """Register compare-sets arguments on *parser*."""
    parser.add_argument(
        "--compare-sets",
        action="store_true",
        default=False,
        help="Compare multiple checkpoints against the first positional checkpoint.",
    )
    parser.add_argument(
        "--compare-labels",
        nargs="+",
        metavar="LABEL",
        default=None,
        help="Labels for each candidate checkpoint (positional args 2+).",
    )
    parser.add_argument(
        "--compare-top-n",
        type=int,
        default=None,
        metavar="N",
        help="Show only the top N most-different candidates.",
    )


def apply_compare_sets(
    args: argparse.Namespace,
    checkpoints: List[dict],
    labels: Optional[List[str]] = None,
) -> Optional[str]:
    """Run compare-sets if the flag is set; return formatted report or None."""
    if not getattr(args, "compare_sets", False):
        return None
    if len(checkpoints) < 2:
        return "compare-sets requires at least two checkpoints."

    from checkpoint_diff.compare_sets import compare_against_reference

    reference = checkpoints[0]
    ref_label = (labels or ["reference"])[0]
    candidate_labels = getattr(args, "compare_labels", None) or [
        (labels[i] if labels and i < len(labels) else f"candidate-{i}")
        for i in range(1, len(checkpoints))
    ]
    candidates = dict(zip(candidate_labels, checkpoints[1:]))
    result = compare_against_reference(reference, candidates, reference_label=ref_label)
    top_n = getattr(args, "compare_top_n", None)
    return format_compare_set(result, top_n=top_n)
