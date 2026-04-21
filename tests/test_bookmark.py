"""Tests for checkpoint_diff.bookmark."""

from __future__ import annotations

import json
import pytest

from checkpoint_diff.bookmark import (
    BookmarkStore,
    filter_by_bookmark,
    load_bookmarks,
    save_bookmarks,
)


# ---------------------------------------------------------------------------
# BookmarkStore unit tests
# ---------------------------------------------------------------------------

def test_add_and_get():
    store = BookmarkStore()
    store.add("important", "layer.0.weight")
    assert store.get("important") == ["layer.0.weight"]


def test_add_duplicate_ignored():
    store = BookmarkStore()
    store.add("watch", "bias")
    store.add("watch", "bias")
    assert store.get("watch") == ["bias"]


def test_get_missing_returns_empty():
    store = BookmarkStore()
    assert store.get("nonexistent") == []


def test_remove_existing_key():
    store = BookmarkStore()
    store.add("grp", "k1")
    store.add("grp", "k2")
    removed = store.remove("grp", "k1")
    assert removed is True
    assert store.get("grp") == ["k2"]


def test_remove_last_key_deletes_label():
    store = BookmarkStore()
    store.add("solo", "only")
    store.remove("solo", "only")
    assert "solo" not in store.labels()


def test_remove_missing_key_returns_false():
    store = BookmarkStore()
    assert store.remove("ghost", "x") is False


def test_labels_returns_all():
    store = BookmarkStore()
    store.add("a", "k1")
    store.add("b", "k2")
    assert set(store.labels()) == {"a", "b"}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip(tmp_path):
    store = BookmarkStore()
    store.add("grp", "w1")
    store.add("grp", "w2")
    p = str(tmp_path / "bm.json")
    save_bookmarks(store, p)
    loaded = load_bookmarks(p)
    assert loaded.get("grp") == ["w1", "w2"]


def test_load_invalid_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json")
    with pytest.raises(Exception):
        load_bookmarks(str(p))


def test_load_non_dict_raises(tmp_path):
    p = tmp_path / "list.json"
    p.write_text(json.dumps(["a", "b"]))
    with pytest.raises(ValueError):
        load_bookmarks(str(p))


# ---------------------------------------------------------------------------
# filter_by_bookmark
# ---------------------------------------------------------------------------

def test_filter_by_bookmark_returns_subset():
    store = BookmarkStore()
    store.add("watch", "layer.0.weight")
    diff = {"layer.0.weight": object(), "layer.1.weight": object()}
    result = filter_by_bookmark(diff, store, "watch")
    assert set(result.keys()) == {"layer.0.weight"}


def test_filter_by_bookmark_unknown_label_returns_empty():
    store = BookmarkStore()
    diff = {"k": object()}
    assert filter_by_bookmark(diff, store, "missing") == {}
