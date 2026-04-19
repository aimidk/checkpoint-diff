# Checkpoint Patching

The `checkpoint_diff.patch` module lets you **apply a diff as a patch** to an
existing checkpoint, producing a new checkpoint that reflects the changes
captured in the diff.

## Basic Usage

```python
from checkpoint_diff.diff import compute_diff
from checkpoint_diff.patch import apply_patch, patch_summary

diff = compute_diff(base_ckpt, new_ckpt)
patched = apply_patch(base_ckpt, diff)
```

`apply_patch` returns a plain `dict[str, np.ndarray]` that can be saved with
your preferred serialisation method (e.g. `np.savez`).

## Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `skip_added` | `False` | Exclude keys that only appear in the newer checkpoint. |
| `skip_removed` | `False` | Retain keys that were deleted in the newer checkpoint. |

## Patch Summary

A concise human-readable description of what the patch will do:

```python
print(patch_summary(diff))
# Patch: 3 changed, 1 added, 2 removed, 10 unchanged
```

## Notes

- `apply_patch` never modifies the *base* checkpoint in-place; all arrays are
  copied.
- For *changed* tensors the value from the **newer** checkpoint (`b`) is used.
- Shape mismatches are handled transparently — the newer tensor is used as-is.
