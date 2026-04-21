"""Filtering utilities for CheckpointDiff results."""

from __future__ import annotations

from typing import Iterable

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


def filter_by_status(
    diff: CheckpointDiff,
    *,
    include_added: bool = True,
    include_removed: bool = True,
    include_changed: bool = True,
    include_unchanged: bool = True,
) -> CheckpointDiff:
    """Return a new CheckpointDiff containing only entries matching the given statuses."""
    kept: dict[str, TensorDiff] = {}
    for key, td in diff.items():
        if td.status == "added" and not include_added:
            continue
        if td.status == "removed" and not include_removed:
            continue
        if td.status == "changed" and not include_changed:
            continue
        if td.status == "unchanged" and not include_unchanged:
            continue
        kept[key] = td
    return kept


def filter_by_key_pattern(diff: CheckpointDiff, patterns: Iterable[str]) -> CheckpointDiff:
    """Return entries whose keys match ANY of the given glob-style patterns."""
    import fnmatch

    patterns = list(patterns)
    if not patterns:
        return dict(diff)
    return {
        key: td
        for key, td in diff.items()
        if any(fnmatch.fnmatch(key, p) for p in patterns)
    }


def filter_by_max_abs_mean(
    diff: CheckpointDiff, threshold: float
) -> CheckpointDiff:
    """Keep only 'changed' entries whose abs mean difference exceeds *threshold*.

    Non-changed entries are always kept.
    """
    result: CheckpointDiff = {}
    for key, td in diff.items():
        if td.status == "changed" and td.mean_diff is not None:
            if abs(td.mean_diff) < threshold:
                continue
        result[key] = td
    return result


def filter_by_key_prefix(diff: CheckpointDiff, prefixes: Iterable[str]) -> CheckpointDiff:
    """Return entries whose keys start with ANY of the given prefixes.

    This is a faster alternative to :func:`filter_by_key_pattern` when you only
    need simple prefix matching (e.g. ``"encoder."`` or ``"decoder."``).

    Args:
        diff: The :class:`~checkpoint_diff.diff.CheckpointDiff` to filter.
        prefixes: An iterable of prefix strings to match against.

    Returns:
        A new ``CheckpointDiff`` containing only the matching entries.
    """
    prefix_list = list(prefixes)
    if not prefix_list:
        return dict(diff)
    return {
        key: td
        for key, td in diff.items()
        if any(key.startswith(p) for p in prefix_list)
    }
