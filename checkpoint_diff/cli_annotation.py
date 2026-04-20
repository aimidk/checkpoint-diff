"""CLI helpers for the annotation feature."""
from __future__ import annotations

import argparse
from pathlib import Path

from checkpoint_diff.annotation import (
    AnnotationStore,
    annotate_report,
    load_annotations,
    save_annotations,
)


def add_annotation_args(parser: argparse.ArgumentParser) -> None:
    """Attach annotation-related arguments to *parser*."""
    parser.add_argument(
        "--annotations",
        metavar="FILE",
        default=None,
        help="Path to a JSON annotations file to load and display.",
    )
    parser.add_argument(
        "--annotate",
        nargs=2,
        metavar=("KEY", "NOTE"),
        action="append",
        default=[],
        help="Add an annotation (may be repeated).",
    )
    parser.add_argument(
        "--save-annotations",
        metavar="FILE",
        default=None,
        dest="save_annotations",
        help="Save the resulting annotations to a JSON file.",
    )


def apply_annotations(args: argparse.Namespace, diff) -> str | None:
    """Load, update, optionally save, and render annotations for *diff*.

    Returns a formatted string if an annotations file is involved, else None.
    """
    if not args.annotations and not args.annotate:
        return None

    if args.annotations and Path(args.annotations).exists():
        store = load_annotations(args.annotations)
    else:
        store = AnnotationStore()

    for key, note in args.annotate:
        store.add(key, note)

    if args.save_annotations:
        save_annotations(store, args.save_annotations)

    return annotate_report(diff, store)
