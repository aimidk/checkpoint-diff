# Annotations

The annotation feature lets you attach human-readable notes to specific tensor
keys in a diff, making it easy to document why a weight changed or flag
suspicious layers for later review.

## File format

Annotations are stored as a plain JSON file:

```json
{
  "notes": {
    "encoder.layer.0.weight": "intentional re-init",
    "decoder.bias": "gradient explosion suspected"
  }
}
```

## CLI usage

```bash
# Load an existing annotations file and display notes alongside the diff
checkpoint-diff a.npz b.npz --annotations my_notes.json

# Add a new annotation on the fly
checkpoint-diff a.npz b.npz \
  --annotate "encoder.layer.0.weight" "intentional re-init"

# Combine loading + adding new notes, then persist the result
checkpoint-diff a.npz b.npz \
  --annotations my_notes.json \
  --annotate "decoder.bias" "gradient explosion suspected" \
  --save-annotations updated_notes.json
```

## Python API

```python
from checkpoint_diff.annotation import AnnotationStore, annotate_report
from checkpoint_diff.annotation import load_annotations, save_annotations

store = AnnotationStore()
store.add("fc.weight", "large shift after epoch 10")

# Render notes alongside a CheckpointDiff
print(annotate_report(diff, store))

# Persist for later
save_annotations(store, "notes.json")

# Reload
store2 = load_annotations("notes.json")
```

## Output example

```
fc.bias   [unchanged]
fc.weight [changed]  # large shift after epoch 10
```
