"""CLI helpers for the --similarity flag."""
from __future__ import annotations

import argparse

from .similarity import compute_similarity, format_similarity
from .diff import CheckpointDiff


def add_similarity_args(parser: argparse.ArgumentParser) -> None:
    """Register similarity-related arguments onto *parser*."""
    parser.add_argument(
        "--similarity",
        action="store_true",
        default=False,
        help="Print cosine and L2-norm similarity metrics for changed tensors.",
    )
    parser.add_argument(
        "--similarity-export",
        metavar="FILE",
        default=None,
        help="Write similarity report to FILE (plain text).",
    )


def apply_similarity(
    args: argparse.Namespace,
    diff: CheckpointDiff,
) -> None:
    """Compute and display/export similarity if requested."""
    if not args.similarity and args.similarity_export is None:
        return

    report = compute_similarity(diff)
    text = format_similarity(report)

    if args.similarity:
        print("\n--- Similarity Report ---")
        print(text)

    if args.similarity_export:
        with open(args.similarity_export, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
