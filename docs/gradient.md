# Gradient Norm Analysis

The `--gradient` flag computes and displays the **L2 norm** of each tensor in
both checkpoints, along with the absolute delta and relative percentage change.

This is useful for diagnosing training instabilities or verifying that a
checkpoint update had the expected magnitude of change.

## Usage

```bash
checkpoint-diff model_a.npz model_b.npz --gradient
```

Sample output:

```
Key                                          L2(A)        L2(B)        Delta     Rel%
------------------------------------------------------------------------------------
layer.weight                              2.345678     3.891234    +1.545556  +65.89
layer.bias                                0.123456     0.130000    +0.006544   +5.30
```

## Options

| Flag | Description |
|------|-------------|
| `--gradient` | Enable gradient norm analysis |
| `--gradient-top-n N` | Show only the top *N* keys by absolute norm delta |
| `--gradient-threshold T` | Only show keys whose `|delta|` exceeds *T* |

## Interpretation

- **L2(A)** / **L2(B)**: Euclidean norm of the flattened tensor in each checkpoint.
- **Delta**: `L2(B) - L2(A)` — positive means the tensor grew in magnitude.
- **Rel%**: Relative change as a percentage of `L2(A)`.

Keys that are *added* (only present in B) show `nan` for `L2(A)`, and keys
that are *removed* show `nan` for `L2(B)`.

## Combining with other flags

Gradient norm analysis can be combined with `--filter`, `--top-n`, and
`--export` flags for richer reporting workflows.
