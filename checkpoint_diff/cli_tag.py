"""CLI helpers for the tagging feature."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.tag import TagStore, filter_diff_by_tag, format_tags
from checkpoint_diff.diff import CheckpointDiff


def add_tag_args(parser: argparse.ArgumentParser) -> None:
    """Register tag-related arguments on *parser*."""
    grp = parser.add_argument_group("tagging")
    grp.add_argument(
        "--tag",
        metavar="KEY:TAG",
        action="append",
        default=[],
        help="Attach TAG to KEY (repeatable). Format: layer.weight:frozen",
    )
    grp.add_argument(
        "--filter-tag",
        metavar="TAG",
        default=None,
        help="Only show tensors that have this tag.",
    )
    grp.add_argument(
        "--show-tags",
        action="store_true",
        default=False,
        help="Print tag table after the diff report.",
    )


def apply_tags(
    args: argparse.Namespace,
    diff: CheckpointDiff,
    store: Optional[TagStore] = None,
) -> tuple[CheckpointDiff, TagStore]:
    """Parse tag args, populate *store*, and optionally filter *diff*."""
    if store is None:
        store = TagStore()

    for item in getattr(args, "tag", []):
        if ":" not in item:
            raise ValueError(f"--tag value must be KEY:TAG, got: {item!r}")
        key, tag = item.split(":", 1)
        store.add(key.strip(), tag.strip())

    filter_tag: Optional[str] = getattr(args, "filter_tag", None)
    if filter_tag:
        diff = filter_diff_by_tag(diff, store, filter_tag)

    if getattr(args, "show_tags", False):
        all_keys = (
            list(diff.added) + list(diff.removed)
            + list(diff.changed) + list(diff.unchanged)
        )
        print("\n--- Tags ---")
        print(format_tags(store, sorted(set(all_keys))))

    return diff, store
