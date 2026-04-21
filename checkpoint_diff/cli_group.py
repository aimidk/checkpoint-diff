"""CLI helpers for the grouping feature."""
from __future__ import annotations

import argparse
import json
from typing import Dict, Optional

from checkpoint_diff.group import (
    GroupResult,
    format_groups,
    group_by_map,
    group_by_prefix,
)


def add_group_args(parser: argparse.ArgumentParser) -> None:
    """Register grouping-related arguments on *parser*."""
    grp = parser.add_argument_group("grouping")
    grp.add_argument(
        "--group-by-prefix",
        dest="group_by_prefix",
        action="store_true",
        default=False,
        help="Group diff keys by their first dot-separated component.",
    )
    grp.add_argument(
        "--group-sep",
        dest="group_sep",
        default=".",
        metavar="SEP",
        help="Separator used when grouping by prefix (default: '.').",
    )
    grp.add_argument(
        "--group-map",
        dest="group_map",
        default=None,
        metavar="JSON",
        help="JSON string mapping key names to group labels.",
    )


def apply_grouping(args: argparse.Namespace, diff) -> Optional[Dict[str, GroupResult]]:
    """Apply grouping based on parsed *args*; print and return groups or None."""
    groups = None
    if getattr(args, "group_map", None):
        mapping: Dict[str, str] = json.loads(args.group_map)
        groups = group_by_map(diff, mapping)
    elif getattr(args, "group_by_prefix", False):
        sep = getattr(args, "group_sep", ".")
        groups = group_by_prefix(diff, sep=sep)

    if groups is not None:
        print(format_groups(groups))
    return groups
