# Regression Detection

`checkpoint-diff` can flag tensors that have moved *away* from a known-good
reference checkpoint, helping you catch regressions during iterative training.

## Concept

For each tensor that changed between checkpoint **A** and checkpoint **B**, the
regression detector measures:

- **dist_a** = |mean(A) − mean(ref)|
- **dist_b** = |mean(B) − mean(ref)|

If `dist_b > dist_a + tolerance`, the tensor is flagged as **regressed**.

The `direction` field tells you at a glance whether a tensor moved *toward* or
*away* from the reference, or stayed *neutral* (within tolerance).

## CLI Usage

```bash
checkpoint-diff model_v1.npz model_v2.npz \
    --regression-ref golden.npz
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--regression-ref PATH` | `None` | Path to the reference checkpoint |
| `--regression-tolerance TOL` | `0.0` | Allowed increase in mean-distance before flagging |
| `--regression-show-all` | off | Print all compared tensors, not just regressions |

## Example Output

```
Key                                          MeanA      MeanB        Ref Dir
----------------------------------------------------------------------------------
layer1.weight                               1.0000     5.0000     0.0000 away

1 regression(s) detected out of 2 compared.
```

## Python API

```python
from checkpoint_diff.loader import load_checkpoint
from checkpoint_diff.diff import compute_diff
from checkpoint_diff.regression import detect_regression, format_regression

a = load_checkpoint("model_v1.npz")
b = load_checkpoint("model_v2.npz")
ref = load_checkpoint("golden.npz")

diff = compute_diff(a, b)
report = detect_regression(diff, ref, tolerance=0.01)
print(format_regression(report))
```

## Notes

- Only tensors with status `changed` or `added` are evaluated.
- Tensors absent from the reference checkpoint are silently skipped.
- Use `--regression-show-all` during debugging to see every compared tensor.
