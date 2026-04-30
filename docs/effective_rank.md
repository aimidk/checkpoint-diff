# Effective Rank

The **effective rank** of a weight tensor measures how many singular values
contribute meaningfully to the matrix, based on the entropy of the normalised
singular-value spectrum (Roy & Vetterli, 2007).

## Formula

Given the singular values `σ₁ ≥ σ₂ ≥ … ≥ σₙ > 0`:

```
p_i  = σ_i / Σ σ_j
H    = -Σ p_i · ln(p_i)   (Shannon entropy)
rank = exp(H)
```

A **rank of 1** means the tensor is effectively rank-one (one dominant direction).
A **rank equal to min(rows, cols)** means all singular values are equal — maximum
distribution of information.

## CLI Usage

```bash
checkpoint-diff a.npz b.npz --effective-rank
checkpoint-diff a.npz b.npz --effective-rank --effective-rank-top-n 10
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--effective-rank` | off | Enable effective rank analysis |
| `--effective-rank-top-n N` | all rows | Show only top N rows by \|delta\| |

## Output

```
Key                                      RankA    RankB    Delta  Status
------------------------------------------------------------------------
layer1.weight                            3.981    1.000   -2.981  changed
layer2.weight                            2.000    2.000    0.000  unchanged
```

## Interpretation

- A **large negative delta** means the layer collapsed toward a lower-rank
  representation — potentially a sign of over-fitting or gradient vanishing.
- A **large positive delta** means the layer spread its representation across
  more directions — often healthy during early training.
- Tensors with fewer than 2 dimensions are reshaped to a row vector before SVD.
