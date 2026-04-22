"""CLI integration for the diff-score feature."""
from __future__ import annotations

import argparse
import json

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.diff_score import DiffScore, compute_score, format_score


def add_diff_score_args(parser: argparse.ArgumentParser) -> None:
    """Register --score and --score-json flags on *parser*."""
    group = parser.add_argument_group("diff score")
    group.add_argument(
        "--score",
        action="store_true",
        default=False,
        help="Print a scalar diff score summarising overall divergence.",
    )
    group.add_argument(
        "--score-json",
        action="store_true",
        default=False,
        help="Emit the diff score as a JSON object (implies --score).",
    )


def apply_diff_score(
    args: argparse.Namespace,
    diff: CheckpointDiff,
) -> str | None:
    """Compute and return the diff-score output string, or None if not requested.

    Returns a formatted string when ``--score`` or ``--score-json`` is set,
    otherwise returns *None*.
    """
    if not (getattr(args, "score", False) or getattr(args, "score_json", False)):
        return None

    ds: DiffScore = compute_score(diff)

    if getattr(args, "score_json", False):
        return json.dumps(
            {
                "score": ds.score,
                "total_keys": ds.total_keys,
                "changed_keys": ds.changed_keys,
                "added_keys": ds.added_keys,
                "removed_keys": ds.removed_keys,
                "mean_abs_delta": ds.mean_abs_delta,
                "max_abs_delta": ds.max_abs_delta,
            },
            indent=2,
        )

    return format_score(ds)
