"""CLI helpers for the bookmark feature."""

from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.bookmark import BookmarkStore, filter_by_bookmark


def add_bookmark_args(parser: argparse.ArgumentParser) -> None:
    """Register bookmark-related arguments on *parser*."""
    grp = parser.add_argument_group("bookmarks")
    grp.add_argument(
        "--bookmark-file",
        metavar="FILE",
        default=None,
        help="Path to JSON bookmark store.",
    )
    grp.add_argument(
        "--bookmark-label",
        metavar="LABEL",
        default=None,
        help="Filter diff output to keys bookmarked under LABEL.",
    )
    grp.add_argument(
        "--list-bookmarks",
        action="store_true",
        default=False,
        help="Print all bookmark labels in the store and exit.",
    )


def apply_bookmarks(
    args: argparse.Namespace,
    diff: dict,
) -> Optional[dict]:
    """Apply bookmark filtering to *diff* based on parsed *args*.

    Returns filtered diff or original diff if no bookmark args active.
    """
    if not args.bookmark_file:
        return diff

    from checkpoint_diff.bookmark import load_bookmarks

    store = load_bookmarks(args.bookmark_file)

    if args.list_bookmarks:
        for label in store.labels():
            keys = store.get(label)
            print(f"{label}: {', '.join(keys)}")
        return None  # signal caller to exit

    if args.bookmark_label:
        return filter_by_bookmark(diff, store, args.bookmark_label)

    return diff
