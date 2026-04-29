"""CLI integration for sign-flip detection."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.sign_flip import compute_sign_flips, format_sign_flips


def add_sign_flip_args(parser: argparse.ArgumentParser) -> None:
    """Register sign-flip flags on *parser*."""
    grp = parser.add_argument_group("sign-flip analysis")
    grp.add_argument(
        "--sign-flips",
        action="store_true",
        default=False,
        help="Report fraction of weights that changed sign between checkpoints.",
    )
    grp.add_argument(
        "--sign-flip-min-rate",
        type=float,
        default=0.0,
        metavar="RATE",
        help="Only report keys whose flip rate is at least RATE (0–1). Default: 0.",
    )
    grp.add_argument(
        "--sign-flip-top-n",
        type=int,
        default=None,
        metavar="N",
        help="Limit sign-flip output to the top-N keys by flip rate.",
    )


def apply_sign_flip(
    args: argparse.Namespace,
    diff: CheckpointDiff,
) -> Optional[str]:
    """Return formatted sign-flip report if the flag is set, else None."""
    if not getattr(args, "sign_flips", False):
        return None
    rows = compute_sign_flips(
        diff,
        min_flip_rate=getattr(args, "sign_flip_min_rate", 0.0),
    )
    return format_sign_flips(rows, top_n=getattr(args, "sign_flip_top_n", None))
