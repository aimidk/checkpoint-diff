"""CLI integration for multi-step history analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .history import load_history, compute_trends
from .history_report import print_trend_report, export_trends_csv


def add_history_args(parser: argparse.ArgumentParser) -> None:
    """Register history sub-command arguments onto *parser*."""
    parser.add_argument(
        "checkpoints",
        nargs="+",
        metavar="CHECKPOINT",
        help="Ordered list of checkpoint files (oldest → newest).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        metavar="N",
        help="Show the top-N most-changed keys (default: 20).",
    )
    parser.add_argument(
        "--export-csv",
        metavar="FILE",
        default=None,
        help="Export trend table to a CSV file.",
    )
    parser.add_argument(
        "--prefix",
        metavar="PREFIX",
        default=None,
        help="Strip a common key prefix before analysis.",
    )


def run_history(args: argparse.Namespace) -> int:
    """Execute the history sub-command and return an exit code.

    Parameters
    ----------
    args:
        Parsed namespace produced by :func:`add_history_args`.

    Returns
    -------
    int
        0 on success, non-zero on error.
    """
    paths: List[str] = args.checkpoints
    if len(paths) < 2:
        print("error: at least two checkpoints are required for history analysis.")
        return 1

    try:
        step_diffs = load_history(paths, strip_prefix=args.prefix)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1
    except ValueError as exc:
        print(f"error: {exc}")
        return 1

    trends = compute_trends(step_diffs)

    if args.export_csv:
        dest = Path(args.export_csv)
        try:
            export_trends_csv(trends, dest)
            print(f"Trends exported to {dest}")
        except OSError as exc:
            print(f"error: could not write CSV — {exc}")
            return 1

    print_trend_report(trends, top_n=args.top)
    return 0
