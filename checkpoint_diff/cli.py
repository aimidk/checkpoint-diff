"""CLI entry point for checkpoint-diff."""
from __future__ import annotations

import argparse
import sys

from checkpoint_diff.loader import load_checkpoint
from checkpoint_diff.diff import compute_diff, has_differences
from checkpoint_diff.filter import filter_by_status, filter_by_key_pattern, filter_by_max_abs_mean
from checkpoint_diff.report import print_report
from checkpoint_diff.export import export_diff


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="checkpoint-diff",
        description="Compare two ML model checkpoints and visualize differences.",
    )
    p.add_argument("checkpoint_a", help="Path to the first checkpoint file.")
    p.add_argument("checkpoint_b", help="Path to the second checkpoint file.")
    p.add_argument(
        "--exit-code",
        action="store_true",
        help="Exit with code 1 when differences are found.",
    )
    p.add_argument(
        "--show",
        choices=["all", "changed", "added", "removed", "unchanged"],
        default="all",
        help="Filter which tensor statuses to display.",
    )
    p.add_argument(
        "--pattern",
        default=None,
        metavar="GLOB",
        help="Only show keys matching this glob pattern.",
    )
    p.add_argument(
        "--max-abs-mean",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="Only show entries whose mean_abs_diff exceeds THRESHOLD.",
    )
    p.add_argument(
        "--export",
        choices=["json", "csv"],
        default=None,
        help="Export the diff to stdout in the given format instead of the default report.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    ckpt_a = load_checkpoint(args.checkpoint_a)
    ckpt_b = load_checkpoint(args.checkpoint_b)
    diff = compute_diff(ckpt_a, ckpt_b)

    if args.show != "all":
        statuses = {args.show}
        diff = filter_by_status(diff, statuses)
    if args.pattern:
        diff = filter_by_key_pattern(diff, args.pattern)
    if args.max_abs_mean is not None:
        diff = filter_by_max_abs_mean(diff, args.max_abs_mean)

    if args.export:
        print(export_diff(diff, args.export))
    else:
        print_report(diff)

    if args.exit_code and has_differences(diff):
        sys.exit(1)


if __name__ == "__main__":
    main()
