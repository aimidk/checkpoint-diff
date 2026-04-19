# checkpoint-diff

A CLI tool to compare and visualize diffs between ML model checkpoints.

## Installation

```bash
pip install checkpoint-diff
```

## Usage

Compare two model checkpoints and visualize the differences:

```bash
checkpoint-diff compare model_v1.pt model_v2.pt
```

Show a summary of weight changes between checkpoints:

```bash
checkpoint-diff compare model_v1.pt model_v2.pt --summary
```

Export the diff to a file:

```bash
checkpoint-diff compare model_v1.pt model_v2.pt --output diff_report.json
```

Filter to only show layers with significant changes:

```bash
checkpoint-diff compare model_v1.pt model_v2.pt --threshold 0.01
```

### Options

| Flag | Description |
|------|-------------|
| `--summary` | Print a high-level summary of changes |
| `--output` | Save the diff report to a file |
| `--threshold` | Highlight parameters with changes above a given value |
| `--format` | Output format: `table`, `json`, or `csv` |

### Example Output

```
Layer                  | Shape       | Mean Δ     | Max Δ
-----------------------|-------------|------------|----------
encoder.layer1.weight  | [768, 768]  | 0.0023     | 0.1847
encoder.layer2.bias    | [768]       | 0.0001     | 0.0312
decoder.output.weight  | [768, 1000] | 0.0089     | 0.4521
```

## Requirements

- Python 3.8+
- PyTorch or TensorFlow (optional, based on checkpoint format)

## License

This project is licensed under the [MIT License](LICENSE).
