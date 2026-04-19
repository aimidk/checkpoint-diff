"""CLI helpers that wire --prefix-a / --prefix-b / --auto-align into main."""
from __future__ import annotations

import argparse
from typing import Dict

import numpy as np

from checkpoint_diff.align import align_checkpoints


def add_align_args(parser: argparse.ArgumentParser) -> None:
    """Attach alignment arguments to an existing argument parser."""
    grp = parser.add_argument_group("key alignment")
    grp.add_argument(
        "--prefix-a",
        default="",
        metavar="PREFIX",
        help="Strip this prefix from keys in CHECKPOINT_A before comparing.",
    )
    grp.add_argument(
        "--prefix-b",
        default="",
        metavar="PREFIX",
        help="Strip this prefix from keys in CHECKPOINT_B before comparing.",
    )
    grp.add_argument(
        "--auto-align",
        action="store_true",
        help="Auto-detect and strip differing common prefixes.",
    )


def apply_alignment(
    args: argparse.Namespace,
    ckpt_a: Dict[str, np.ndarray],
    ckpt_b: Dict[str, np.ndarray],
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Return aligned checkpoints according to parsed CLI args."""
    return align_checkpoints(
        ckpt_a,
        ckpt_b,
        prefix_a=getattr(args, "prefix_a", ""),
        prefix_b=getattr(args, "prefix_b", ""),
        auto_align=getattr(args, "auto_align", False),
    )
