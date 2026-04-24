# Compare Sets

The **compare-sets** feature lets you diff multiple candidate checkpoints
against a single reference checkpoint in one pass and see them ranked by
how much they diverge from the reference.

## Usage

```bash
checkpoint-diff ref.npz cand_a.npz cand_b.npz cand_c.npz \
    --compare-sets \
    --compare-labels cand_a cand_b cand_c \
    --compare-top-n 2
```

The first positional argument is always treated as the **reference**.
All subsequent positional arguments are treated as **candidates**.

## Options

| Flag | Default | Description |
|---|---|---|
| `--compare-sets` | off | Enable multi-checkpoint comparison mode. |
| `--compare-labels LABEL …` | auto | Human-readable labels for each candidate. |
| `--compare-top-n N` | all | Show only the top *N* most-different candidates. |

## Output

The report is a ranked table:

```
Comparison against reference: ref
------------------------------------------------------------
  Label        Score    Changed   Added   Removed
------------------------------------------------------------
  cand_c      1.2345          3       0         1
  cand_a      0.4321          1       0         0
------------------------------------------------------------
```

- **Score** – overall diff score (higher = more different).
- **Changed / Added / Removed** – raw tensor counts.

## Python API

```python
from checkpoint_diff.compare_sets import compare_against_reference, format_compare_set

result = compare_against_reference(
    reference=ref_ckpt,
    candidates={"run_a": ckpt_a, "run_b": ckpt_b},
    reference_label="baseline",
)
print(format_compare_set(result, top_n=5))
```
