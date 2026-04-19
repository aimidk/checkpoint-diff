# Threshold Flagging

`checkpoint-diff` can flag tensors whose statistics change beyond configurable thresholds, making it easy to catch unexpectedly large weight shifts during training or fine-tuning.

## Supported Thresholds

| Option | Description |
|---|---|
| `max_mean_delta` | Maximum allowed absolute change in mean |
| `max_std_delta` | Maximum allowed absolute change in standard deviation |
| `max_max_delta` | Maximum allowed absolute change in max value |

## Python API

```python
from checkpoint_diff.threshold import ThresholdConfig, flag_tensors, format_flagged
from checkpoint_diff.diff import compute_diff
from checkpoint_diff.loader import load_checkpoint

ckpt_a = load_checkpoint("model_epoch1.pt")
ckpt_b = load_checkpoint("model_epoch2.pt")
diff = compute_diff(ckpt_a, ckpt_b)

cfg = ThresholdConfig(max_mean_delta=0.01, max_std_delta=0.05)
flagged = flag_tensors(diff, cfg)
print(format_flagged(flagged))
```

## Output Example

```
Flagged tensors:
  encoder.layer.0.weight: mean_delta=0.034 (threshold=0.01)
  decoder.bias: std_delta=0.12 (threshold=0.05)
```

If no tensors exceed any threshold:

```
No tensors exceeded thresholds.
```

## Notes

- Only `changed` tensors are evaluated; `added` and `removed` keys are ignored.
- Multiple thresholds can be combined in a single `ThresholdConfig`.
- Each exceeded threshold produces a separate `FlaggedTensor` entry.
