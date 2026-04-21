# Key Rename Mapping

The rename feature lets you apply a key-name translation before comparing two
checkpoints. This is useful when two models share the same weights but use
different naming conventions (e.g. `layer.0.weight` vs `encoder.0.weight`).

## Rename map format

A rename map is a plain JSON object stored in a `.json` file:

```json
{
  "layer.0.weight": "encoder.0.weight",
  "layer.0.bias":   "encoder.0.bias",
  "head.weight":    "classifier.weight"
}
```

Each key is an *old* name (as it appears in the checkpoint) and each value is
the *new* name to use during the diff.

## Python API

```python
from checkpoint_diff.rename import load_rename_map, apply_rename_map

rename_map = load_rename_map("rename.json")

# Apply to one or both checkpoints before calling compute_diff
ckpt_a = apply_rename_map(raw_ckpt_a, rename_map)
```

### `load_rename_map(path)`

Loads and validates a JSON rename map from *path*.  Raises `ValueError` if the
file is not a JSON object or contains non-string keys/values.

### `apply_rename_map(checkpoint, rename_map, *, strict=False)`

Returns a **new** dictionary with keys renamed according to *rename_map*.
Keys absent from the map are passed through unchanged.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strict`  | `False` | Raise `KeyError` if a key listed in the map is missing from the checkpoint. |

### `invert_rename_map(rename_map)`

Returns the inverse mapping `{new_key: old_key}`.  Raises `ValueError` if the
map is not bijective (two old keys map to the same new key).

## CLI usage

Pass `--rename-a` and/or `--rename-b` to supply rename maps for each
checkpoint:

```bash
checkpoint-diff model_v1.npz model_v2.npz \
  --rename-a rename_v1.json \
  --rename-b rename_v2.json
```

> **Note:** Rename maps are applied *before* any prefix stripping or alignment.
