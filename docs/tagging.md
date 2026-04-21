# Tensor Tagging

The tagging feature lets you attach human-readable labels to individual tensor
keys in a checkpoint diff. Tags are useful for tracking which layers are frozen,
experimental, or require special review.

## CLI Usage

```bash
# Attach a tag to a specific key
checkpoint-diff a.npz b.npz --tag layer.weight:frozen --tag layer.bias:frozen

# Only show tensors that carry a particular tag
checkpoint-diff a.npz b.npz --tag layer.weight:frozen --filter-tag frozen

# Print the tag table below the diff report
checkpoint-diff a.npz b.npz --tag layer.weight:frozen --show-tags
```

## Python API

```python
from checkpoint_diff.tag import TagStore, filter_diff_by_tag, format_tags

store = TagStore()
store.add("layer.weight", "frozen")
store.add("layer.bias", "frozen")
store.add("head.weight", "trainable")

# Query
print(store.get("layer.weight"))       # ['frozen']
print(store.keys_with_tag("frozen"))   # ['layer.weight', 'layer.bias']
print(store.all_tags())                # ['frozen', 'trainable']

# Filter a diff to only frozen tensors
filtered = filter_diff_by_tag(diff, store, "frozen")

# Pretty-print the tag table
print(format_tags(store))
```

## Tag Format

When using the CLI, each `--tag` argument must follow the `KEY:TAG` format:

```
--tag transformer.layer.0.weight:frozen
```

Multiple tags can be added to the same key by repeating `--tag`.

## Notes

- Tags are **not** persisted between CLI runs unless you integrate `TagStore`
  with `AnnotationStore` (see `docs/annotations.md`).
- Duplicate tags on the same key are silently ignored.
- `--filter-tag` and `--show-tags` can be combined freely.
