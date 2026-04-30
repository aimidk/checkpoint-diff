# Mutual Information Analysis

The `--mutual-info` flag estimates the **pairwise mutual information** between
the two versions of each tensor (checkpoint A vs checkpoint B).

Mutual information (MI) measures how much knowing tensor A tells you about
tensor B. High MI indicates the tensors share significant statistical structure;
near-zero MI suggests they have diverged substantially.

## Usage

```bash
checkpoint-diff model_v1.npz model_v2.npz --mutual-info
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--mutual-info` | off | Enable mutual information analysis |
| `--mi-bins N` | 64 | Histogram bins for density estimation |
| `--mi-top-n N` | all | Show only top-N keys by MI |

## Output columns

| Column | Description |
|---|---|
| `Key` | Tensor parameter name |
| `MI (bits)` | Estimated mutual information in bits |
| `H(a)` | Shannon entropy of tensor A (bits) |
| `H(b)` | Shannon entropy of tensor B (bits) |
| `NMI` | Normalised MI: `MI / max(H(a), H(b))` ∈ [0, 1] |

## Interpretation

- **NMI ≈ 1.0** — the two tensor versions are nearly identical in distribution.
- **NMI ≈ 0.0** — the tensors have diverged; little shared information.
- Layers with low NMI after fine-tuning are candidates for closer inspection.

## Notes

- MI is estimated via 2-D joint histograms; accuracy improves with larger
  tensors and more bins.
- Keys present in only one checkpoint (added/removed) are skipped because
  only one array is available.
- For very small tensors (< 50 elements) the histogram estimate may be noisy;
  consider increasing `--mi-bins` or interpreting results cautiously.
