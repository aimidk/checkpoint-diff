"""CLI integration for the tensor-clustering feature."""
from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.cluster import cluster_by_magnitude, format_clusters
from checkpoint_diff.diff import CheckpointDiff


def add_cluster_args(parser: argparse.ArgumentParser) -> None:
    """Register --cluster and related flags on *parser*."""
    parser.add_argument(
        "--cluster",
        action="store_true",
        default=False,
        help="Group tensors into magnitude bands and print a cluster table.",
    )
    parser.add_argument(
        "--cluster-bins",
        type=int,
        default=3,
        metavar="N",
        dest="cluster_bins",
        help="Number of magnitude bands (default: 3).",
    )
    parser.add_argument(
        "--cluster-status",
        nargs="+",
        default=None,
        metavar="STATUS",
        dest="cluster_status",
        help="Limit clustering to tensors with these statuses (default: changed added).",
    )


def apply_clustering(
    args: argparse.Namespace,
    diff: CheckpointDiff,
    out=None,
) -> Optional[str]:
    """If ``--cluster`` was requested, compute and print/return the cluster table.

    Parameters
    ----------
    args:  Parsed namespace from :func:`add_cluster_args`.
    diff:  The computed :class:`CheckpointDiff`.
    out:   File-like object for output; defaults to stdout via ``print``.

    Returns the formatted string, or ``None`` if clustering was not requested.
    """
    if not getattr(args, "cluster", False):
        return None

    n_bins: int = getattr(args, "cluster_bins", 3)
    statuses = getattr(args, "cluster_status", None)

    clusters = cluster_by_magnitude(diff, n_bins=n_bins, statuses=statuses)
    table = format_clusters(clusters)

    if out is None:
        print(table)
    else:
        out.write(table + "\n")

    return table
