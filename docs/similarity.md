# Similarity Metrics

`checkpoint-diff` can compute **cosine similarity** and **L2-norm** metrics
between matching tensors in two checkpoints.

## Usage

```bash
# Print similarity table to stdout
checkpoint-diff a.npz b.npz --similarity

# Export similarity report to a file
checkpoint-diff a.npz b.npz --similarity-export report.txt
```

## Metrics

| Column | Description |
|--------|-------------|
| `Cosine` | Cosine similarity ∈ [−1, 1]. `N/A` if either tensor has zero norm or shapes differ. |
| `L2(a)` | L2 norm of the tensor in checkpoint **A**. |
| `L2(b)` | L2 norm of the tensor in checkpoint **B**. |
| `|ΔL2|` | Absolute difference of L2 norms. |

## Interpretation

- **Cosine ≈ 1.0** — tensors point in the same direction; only magnitude may differ.
- **Cosine ≈ 0.0** — tensors are orthogonal; weights have changed substantially.
- **Cosine ≈ −1.0** — weights have flipped sign (e.g. after a poorly-conditioned update).
- **|ΔL2| large** — the scale of the weights has changed significantly.

## Notes

- Only *changed* tensors (status `changed`) are included. Added/removed keys are skipped.
- Multi-dimensional tensors are flattened before computing cosine similarity.
- Combine with `--threshold` to flag tensors whose cosine similarity drops below a target value.
