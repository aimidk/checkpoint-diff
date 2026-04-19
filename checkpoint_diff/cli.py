"""Command-line interface for checkpoint-diff."""

import argparse
import sys

from checkpoint_diff.loader import load_checkpoint
from checkpoint_diff.diff import compute_diff, has_differences
from checkpoint_diff.report import print_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="checkpoint-diff",
        description="Compare and visualize diffs between ML model checkpoints.",
    )
    parser.add_argument("baseline", help="Path to the baseline checkpoint file.")
    parser.add_argument("updated", help="Path to the updated checkpoint file.")
    parser.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        help="Disable colored output.",
    )
    parser.add_argument(
        "--exit-code",
        action="store_true",
        default=False,
        help="Exit with code 1 if differences are found.",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        metavar="KEY",
        default=None,
        help="Only compare the specified tensor keys.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        baseline = load_checkpoint(args.baseline)
        updated = load_checkpoint(args.updated)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.keys:
        baseline = {k: v for k, v in baseline.items() if k in args.keys}
        updated = {k: v for k, v in updated.items() if k in args.keys}

    diff = compute_diff(baseline, updated)
    print_report(diff, color=not args.no_color)

    if args.exit_code and has_differences(diff):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
