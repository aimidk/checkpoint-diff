# Sign-Flip Analysis

The sign-flip module detects weights that reversed their sign (positive→negative
or negative→positive) between two checkpoints. Large-scale sign flips can
indicate instability, exploding/vanishing gradients, or a learning-rate that is
too high.

## CLI usage

```bash
checkpoint-diff model_v1.npz model_v2.npz --sign-flips
```

Optional flags:

| Flag | Description |
|------|-------------|
| `--sign-flips` | Enable sign-flip report |
| `--sign-flip-min-rate RATE` | Only show keys with flip rate ≥ RATE (0–1) |
| `--sign-flip-top-n N` | Limit output to top-N keys by flip rate |

## Example output

```
Key                                       Flipped    Total    Rate    A+%    B+%
--------------------------------------------------------------------------------
transformer.layer.0.weight                   2048     4096  50.0%  51.2%  48.8%
embedding.weight                              512     8192   6.2%  49.3%  50.7%
```

## Python API

```python
from checkpoint_diff.sign_flip import compute_sign_flips, format_sign_flips
from checkpoint_diff.diff import compute_diff
from checkpoint_diff.loader import load_checkpoint

a = load_checkpoint("model_v1.npz")
b = load_checkpoint("model_v2.npz")
diff = compute_diff(a, b)

rows = compute_sign_flips(diff, min_flip_rate=0.05)
print(format_sign_flips(rows, top_n=10))
```

### `SignFlipRow` fields

| Field | Type | Description |
|-------|------|-------------|
| `key` | `str` | Parameter name |
| `total_elements` | `int` | Number of scalar elements |
| `flipped` | `int` | Elements that changed sign |
| `flip_rate` | `float` | `flipped / total_elements` |
| `a_pos_frac` | `float` | Fraction of positive values in A |
| `b_pos_frac` | `float` | Fraction of positive values in B |

> **Note:** Elements that are exactly zero in either checkpoint are excluded
> from the flip count because a transition from/to zero is not a sign reversal.
