"""CLI integration for drift detection."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.drift import detect_drift, format_drift


def add_drift_args(parser: argparse.ArgumentParser) -> None:
    """Register drift-detection arguments on *parser*."""
    parser.add_argument(
        "--drift",
        action="store_true",
        default=False,
        help="Detect and report statistical drift between checkpoints.",
    )
    parser.add_argument(
        "--drift-mean-threshold",
        type=float,
        default=0.1,
        metavar="FRAC",
        help="Relative mean change threshold to flag a tensor (default: 0.1).",
    )
    parser.add_argument(
        "--drift-std-threshold",
        type=float,
        default=0.1,
        metavar="FRAC",
        help="Relative std change threshold to flag a tensor (default: 0.1).",
    )
    parser.add_argument(
        "--drift-top",
        type=int,
        default=None,
        metavar="N",
        help="Show only the top N drifted tensors.",
    )
    parser.add_argument(
        "--drift-include-unchanged",
        action="store_true",
        default=False,
        help="Include tensors that did not exceed the drift threshold in the report.",
    )


def apply_drift(
    args: argparse.Namespace,
    diff: CheckpointDiff,
) -> Optional[str]:
    """Run drift detection if requested and return a formatted report string.

    Returns *None* when ``--drift`` was not passed.
    """
    if not getattr(args, "drift", False):
        return None

    report = detect_drift(
        diff,
        mean_threshold=args.drift_mean_threshold,
        std_threshold=args.drift_std_threshold,
        include_unchanged=args.drift_include_unchanged,
    )
    return format_drift(report, top_n=args.drift_top)
