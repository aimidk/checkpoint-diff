# Key Alignment

When comparing checkpoints from different training frameworks or after refactoring
model code, parameter keys often differ only by a common prefix (e.g. `module.`
added by `DataParallel`, or `net.` vs no prefix).

`checkpoint-diff` provides first-class support for stripping these prefixes
before the diff is computed.

## CLI flags

| Flag | Description |
|------|-------------|
| `--prefix-a PREFIX` | Strip `PREFIX` from every key in the first checkpoint. |
| `--prefix-b PREFIX` | Strip `PREFIX` from every key in the second checkpoint. |
| `--auto-align` | Automatically detect and strip differing common prefixes. |

## Examples

```bash
# Manually strip "module." added by DataParallel
checkpoint-diff model_dp.pt model_plain.pt --prefix-a "module."

# Let the tool figure it out
checkpoint-diff ckpt_v1.npz ckpt_v2.npz --auto-align
```

## Python API

```python
from checkpoint_diff.align import align_checkpoints

aligned_a, aligned_b = align_checkpoints(
    ckpt_a, ckpt_b, prefix_a="module.", prefix_b=""
)
```

After alignment the two dicts share a common key namespace and can be passed
directly to `compute_diff`.
