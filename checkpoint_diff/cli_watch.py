"""CLI integration for the watch feature."""

from __future__ import annotations

import argparse
from pathlib import Path

from checkpoint_diff.diff import has_differences
from checkpoint_diff.report import format_report
from checkpoint_diff.watch import watch, CheckpointDiff


def add_watch_args(parser: argparse.ArgumentParser) -> None:
    """Register watch-related arguments on *parser*."""
    parser.add_argument(
        "--watch",
        metavar="DIR",
        default=None,
        help="Directory to watch for new checkpoints.",
    )
    parser.add_argument(
        "--watch-interval",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Polling interval in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--watch-max-polls",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N polls (useful for testing).",
    )


def _make_handler(verbose: bool) -> object:
    """Return a callable suitable for passing to watch()."""

    def _on_diff(prev: Path, curr: Path, diff: CheckpointDiff) -> None:
        header = f"\n=== {prev.name} -> {curr.name} ==="
        print(header)
        if not has_differences(diff):
            print("  No differences detected.")
        else:
            if verbose:
                print(format_report(diff))
            else:
                changed = sum(
                    1 for td in diff.changed.values() if td is not None
                )
                added = len(diff.added)
                removed = len(diff.removed)
                print(
                    f"  changed={changed}  added={added}  removed={removed}"
                )

    return _on_diff


def apply_watch(args: argparse.Namespace) -> bool:
    """Run the watch loop if --watch was supplied.

    Returns True if the watch loop was entered (so callers can skip normal
    diff processing), False otherwise.
    """
    if not getattr(args, "watch", None):
        return False

    verbose = getattr(args, "verbose", False)
    handler = _make_handler(verbose)
    watch(
        directory=args.watch,
        on_diff=handler,
        interval=args.watch_interval,
        max_polls=args.watch_max_polls,
    )
    return True
