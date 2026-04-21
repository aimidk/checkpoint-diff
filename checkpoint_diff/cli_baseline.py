"""CLI helpers for baseline management sub-commands."""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from checkpoint_diff.baseline import (
    clear_baseline,
    format_baseline_status,
    get_baseline,
    set_baseline,
)


def add_baseline_args(parser: argparse.ArgumentParser) -> None:
    """Register baseline-related arguments onto *parser*."""
    group = parser.add_argument_group("baseline")
    group.add_argument(
        "--set-baseline",
        metavar="PATH",
        default=None,
        help="Pin PATH as the reference baseline checkpoint.",
    )
    group.add_argument(
        "--clear-baseline",
        action="store_true",
        default=False,
        help="Remove the currently pinned baseline.",
    )
    group.add_argument(
        "--show-baseline",
        action="store_true",
        default=False,
        help="Print the current baseline path and exit.",
    )
    group.add_argument(
        "--baseline-store",
        metavar="FILE",
        default=None,
        help="Path to the baseline store file (default: .checkpoint_baseline.json).",
    )


def apply_baseline(
    args: argparse.Namespace,
    checkpoint_b: Optional[str] = None,
) -> Optional[str]:
    """Handle baseline sub-commands and return the effective 'other' checkpoint path.

    Returns the path that should be used as checkpoint B, or None if a
    baseline action was handled and the caller should exit.
    """
    store = args.baseline_store

    if args.show_baseline:
        print(format_baseline_status(store))
        sys.exit(0)

    if args.clear_baseline:
        removed = clear_baseline(store)
        print("Baseline cleared." if removed else "No baseline was set.")
        sys.exit(0)

    if args.set_baseline:
        set_baseline(args.set_baseline, store)
        print(f"Baseline set to: {args.set_baseline}")
        sys.exit(0)

    # If no explicit checkpoint_b, fall back to the stored baseline.
    if checkpoint_b is None:
        checkpoint_b = get_baseline(store)

    return checkpoint_b
