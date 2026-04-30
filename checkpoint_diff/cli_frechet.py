"""CLI integration for Fréchet distance analysis."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.frechet import compute_frechet, format_frechet


def add_frechet_args(parser: argparse.ArgumentParser) -> None:
    """Register --frechet and related flags on *parser*."""
    parser.add_argument(
        "--frechet",
        action="store_true",
        default=False,
        help="Show Fréchet distance between tensor distributions.",
    )
    parser.add_argument(
        "--frechet-top-n",
        type=int,
        default=None,
        metavar="N",
        dest="frechet_top_n",
        help="Limit output to the N keys with the largest Fréchet distance.",
    )
    parser.add_argument(
        "--frechet-threshold",
        type=float,
        default=None,
        metavar="T",
        dest="frechet_threshold",
        help="Warn if any Fréchet distance exceeds T.",
    )


def apply_frechet(
    args: argparse.Namespace,
    diff: CheckpointDiff,
    *,
    print_fn=print,
) -> Optional[int]:
    """If --frechet is set, compute and print the Fréchet distance table.

    Returns a non-zero exit code if --frechet-threshold is set and any
    distance exceeds it, otherwise returns None.
    """
    if not args.frechet:
        return None

    rows = compute_frechet(diff, top_n=args.frechet_top_n)
    print_fn(format_frechet(rows))

    threshold = args.frechet_threshold
    if threshold is not None:
        exceeded = [r for r in rows if not __import__("math").isnan(r.frechet) and r.frechet > threshold]
        if exceeded:
            print_fn(
                f"\n[WARNING] {len(exceeded)} key(s) exceed Fréchet threshold {threshold}: "
                + ", ".join(r.key for r in exceeded)
            )
            return 1
    return None
