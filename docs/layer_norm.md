# Layer Norm Analysis

The `--layer-norm` flag computes per-layer **L1** and **L2** norms for every
tensor present in either checkpoint, and shows how those norms changed.

## Usage

```bash
checkpoint-diff model_a.npz model_b.npz --layer-norm
```

Sample output:

```
Key                                      Status          L1(a)        L1(b)        L2(a)        L2(b)          ΔL2
----------------------------------------------------------------------------------------------------------------------
layer4.weight                            changed      1.23e+03     1.31e+03      4.42e+02     4.61e+02     1.90e+01
layer3.bias                              changed      2.10e+01     2.15e+01      9.80e+00     9.95e+00     1.50e-01
layer1.weight                            unchanged    5.00e+02     5.00e+02      2.10e+02     2.10e+02     0.00e+00
```

Rows are sorted by `|ΔL2|` in descending order so the most-changed layers
appear first.

## Options

| Flag | Default | Description |
|---|---|---|
| `--layer-norm` | off | Enable layer-norm analysis. |
| `--layer-norm-top-n N` | all | Show only the top *N* rows. |
| `--layer-norm-export FILE` | — | Export the full table as a CSV file. |

## Exported CSV columns

| Column | Description |
|---|---|
| `key` | Tensor name |
| `status` | `changed`, `added`, `removed`, or `unchanged` |
| `l1_a` / `l1_b` | L1 norm in checkpoint A / B |
| `l2_a` / `l2_b` | L2 norm in checkpoint A / B |
| `l1_delta` | `l1_b − l1_a` |
| `l2_delta` | `l2_b − l2_a` |

For `added` tensors `*_a` columns are `nan`; for `removed` tensors `*_b`
columns are `nan`.
