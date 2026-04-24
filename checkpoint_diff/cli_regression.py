"""CLI integration for regression detection."""
from __future__ import annotations

import argparse
from typing import Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.loader import load_checkpoint
from checkpoint_diff.regression import detect_regression, format_regression


def add_regression_args(parser: argparse.ArgumentParser) -> None:
    """Register --regression-ref and related flags on *parser*."""
    parser.add_argument(
        "--regression-ref",
        metavar="PATH",
        default=None,
        help="Path to reference checkpoint for regression detection.",
    )
    parser.add_argument(
        "--regression-tolerance",
        type=float,
        default=0.0,
        metavar="TOL",
        help="Tolerance for mean-distance increase before flagging (default: 0.0).",
    )
    parser.add_argument(
        "--regression-show-all",
        action="store_true",
        default=False,
        help="Show all compared tensors, not just regressions.",
    )


def apply_regression(
    args: argparse.Namespace,
    diff: CheckpointDiff,
) -> Optional[str]:
    """Run regression detection if --regression-ref was supplied.

    Returns a formatted string or *None* if the flag was not set.
    """
    ref_path: Optional[str] = getattr(args, "regression_ref", None)
    if not ref_path:
        return None

    tolerance: float = getattr(args, "regression_tolerance", 0.0)
    show_all: bool = getattr(args, "regression_show_all", False)

    reference = load_checkpoint(ref_path)
    report = detect_regression(diff, reference, tolerance=tolerance)
    return format_regression(report, show_all=show_all)
