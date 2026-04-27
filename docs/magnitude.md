# Magnitude Analysis

The **magnitude** feature computes per-tensor L1 and L2 norms for both
checkpoints and reports the relative L2 change between them.

## Usage

```bash
checkpoint-diff model_a.npz model_b.npz --magnitude
```

Limit output to the top 10 tensors with the largest relative L2 shift:

```bash
checkpoint-diff model_a.npz model_b.npz --magnitude --magnitude-top-n 10
```

Export the full magnitude table to a CSV file:

```bash
checkpoint-diff model_a.npz model_b.npz --magnitude --magnitude-export mag.csv
```

## Output columns

| Column | Description |
|---|---|
| `key` | Tensor name |
| `status` | `changed`, `unchanged`, `added`, or `removed` |
| `L2(a)` | L2 norm of tensor in checkpoint A |
| `L2(b)` | L2 norm of tensor in checkpoint B |
| `rel_chg` | Relative L2 change `(L2b - L2a) / L2a`; `n/a` for added/removed |

Rows are sorted by absolute relative L2 change, descending.

## API

```python
from checkpoint_diff.magnitude import compute_magnitude, format_magnitude
from checkpoint_diff.diff import compute_diff
from checkpoint_diff.loader import load_checkpoint

diff = compute_diff(load_checkpoint("a.npz"), load_checkpoint("b.npz"))
rows = compute_magnitude(diff)
print(format_magnitude(rows, top_n=20))
```

## Notes

- For **added** keys, `L2(a)` is reported as `0.0` and `rel_chg` is `n/a`.
- For **removed** keys, `L2(b)` is reported as `0.0` and `rel_chg` is `n/a`.
- A `rel_chg` of `n/a` is also shown when `L2(a) == 0` to avoid division by zero.
