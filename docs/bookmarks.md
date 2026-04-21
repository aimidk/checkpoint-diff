# Bookmarks

Bookmarks let you label and persist a set of tensor keys you care about,
then quickly filter any diff output to just those keys.

## Concept

A **bookmark store** is a JSON file mapping *label* strings to lists of
tensor key strings:

```json
{
  "critical": ["head.weight", "head.bias"],
  "watch":    ["encoder.layer.0.weight"]
}
```

You can maintain multiple labels in a single file and switch between them
with `--bookmark-label`.

## CLI Usage

```bash
# Filter diff output to keys bookmarked under the "critical" label
checkpoint-diff a.npz b.npz \
  --bookmark-file bookmarks.json \
  --bookmark-label critical

# List all labels stored in the file
checkpoint-diff a.npz b.npz \
  --bookmark-file bookmarks.json \
  --list-bookmarks
```

## Python API

```python
from checkpoint_diff.bookmark import BookmarkStore, save_bookmarks, load_bookmarks

# Build a store programmatically
store = BookmarkStore()
store.add("critical", "head.weight")
store.add("critical", "head.bias")
save_bookmarks(store, "bookmarks.json")

# Load and query
store = load_bookmarks("bookmarks.json")
print(store.get("critical"))  # ['head.weight', 'head.bias']

# Remove an entry
store.remove("critical", "head.bias")
```

## Notes

- Duplicate keys within a label are silently ignored.
- Removing the last key from a label deletes the label entirely.
- `--list-bookmarks` prints labels and exits without producing a diff report.
