"""Tests for checkpoint_diff.tag."""
import numpy as np
import pytest

from checkpoint_diff.diff import CheckpointDiff, TensorDiff
from checkpoint_diff.tag import (
    TagStore,
    filter_diff_by_tag,
    format_tags,
)


def _td(status: str = "changed") -> TensorDiff:
    a = np.array([1.0, 2.0])
    b = np.array([1.5, 2.5])
    return TensorDiff(status=status, shape_a=a.shape, shape_b=b.shape,
                      mean_a=float(a.mean()), mean_b=float(b.mean()),
                      std_a=float(a.std()), std_b=float(b.std()),
                      max_abs_diff=0.5)


def _make_diff() -> CheckpointDiff:
    return CheckpointDiff(
        added={"new.weight": _td("added")},
        removed={"old.bias": _td("removed")},
        changed={"layer.weight": _td("changed"), "layer.bias": _td("changed")},
        unchanged={"embed.weight": _td("unchanged")},
    )


def test_add_and_get():
    store = TagStore()
    store.add("layer.weight", "frozen")
    assert store.get("layer.weight") == ["frozen"]


def test_add_duplicate_ignored():
    store = TagStore()
    store.add("layer.weight", "frozen")
    store.add("layer.weight", "frozen")
    assert store.get("layer.weight") == ["frozen"]


def test_get_missing_returns_empty():
    store = TagStore()
    assert store.get("nonexistent") == []


def test_remove_existing_tag():
    store = TagStore()
    store.add("k", "t")
    result = store.remove("k", "t")
    assert result is True
    assert store.get("k") == []


def test_remove_missing_tag_returns_false():
    store = TagStore()
    assert store.remove("k", "t") is False


def test_all_tags():
    store = TagStore()
    store.add("a", "frozen")
    store.add("b", "trainable")
    store.add("a", "important")
    assert store.all_tags() == ["frozen", "important", "trainable"]


def test_keys_with_tag():
    store = TagStore()
    store.add("layer.weight", "frozen")
    store.add("layer.bias", "frozen")
    store.add("embed.weight", "trainable")
    assert sorted(store.keys_with_tag("frozen")) == ["layer.bias", "layer.weight"]


def test_filter_diff_by_tag_keeps_only_tagged():
    diff = _make_diff()
    store = TagStore()
    store.add("layer.weight", "frozen")
    store.add("new.weight", "frozen")
    result = filter_diff_by_tag(diff, store, "frozen")
    assert "layer.weight" in result.changed
    assert "layer.bias" not in result.changed
    assert "new.weight" in result.added


def test_filter_diff_by_tag_no_matches_empty():
    diff = _make_diff()
    store = TagStore()
    result = filter_diff_by_tag(diff, store, "nonexistent")
    assert not result.added and not result.changed


def test_filter_diff_by_tag_preserves_removed_and_unchanged():
    """Tags on removed/unchanged keys should also be respected by filter."""
    diff = _make_diff()
    store = TagStore()
    store.add("old.bias", "frozen")
    store.add("embed.weight", "frozen")
    result = filter_diff_by_tag(diff, store, "frozen")
    assert "old.bias" in result.removed
    assert "embed.weight" in result.unchanged
    assert not result.added
    assert not result.changed


def test_format_tags_lists_keys():
    store = TagStore()
    store.add("layer.weight", "frozen")
    out = format_tags(store)
    assert "layer.weight" in out
    assert "frozen" in out


def test_format_tags_empty_store():
    store = TagStore()
    out = format_tags(store)
    assert "no tags" in out
