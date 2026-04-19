"""Apply a diff as a patch to produce a modified checkpoint."""
from __future__ import annotations

from typing import Dict

import numpy as np

from .diff import CheckpointDiff

Checkpoint = Dict[str, np.ndarray]


def apply_patch(
    base: Checkpoint,
    diff: CheckpointDiff,
    *,
    skip_added: bool = False,
    skip_removed: bool = False,
) -> Checkpoint:
    """Return a new checkpoint by applying *diff* to *base*.

    Parameters
    ----------
    base:
        The original checkpoint tensors.
    diff:
        A :class:`~checkpoint_diff.diff.CheckpointDiff` produced by
        :func:`~checkpoint_diff.diff.compute_diff`.
    skip_added:
        When *True*, keys that exist only in the *other* checkpoint are not
        included in the result.
    skip_removed:
        When *True*, keys that were removed in *other* are kept from *base*.
    """
    result: Checkpoint = {}

    for key, td in diff.tensors.items():
        if td.status == "unchanged":
            result[key] = base[key].copy()
        elif td.status == "changed":
            # Use the *other* (new) tensor when available.
            if td.b is not None:
                result[key] = td.b.copy()
            else:
                result[key] = base[key].copy()
        elif td.status == "added":
            if not skip_added and td.b is not None:
                result[key] = td.b.copy()
        elif td.status == "removed":
            if skip_removed and td.a is not None:
                result[key] = td.a.copy()
            # else: drop the key

    return result


def patch_summary(diff: CheckpointDiff) -> str:
    """Return a human-readable one-liner describing what a patch would do."""
    counts: Dict[str, int] = {"added": 0, "removed": 0, "changed": 0, "unchanged": 0}
    for td in diff.tensors.values():
        counts[td.status] = counts.get(td.status, 0) + 1
    parts = [f"{v} {k}" for k, v in counts.items() if v]
    return "Patch: " + ", ".join(parts)
