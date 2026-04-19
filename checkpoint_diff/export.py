"""Export checkpoint diffs to JSON or CSV formats."""
from __future__ import annotations

import csv
import json
import io
from typing import Union

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


def _tensor_diff_to_dict(key: str, td: TensorDiff) -> dict:
    entry = {"key": key, "status": td.status}
    if td.shape_a is not None:
        entry["shape_a"] = list(td.shape_a)
    if td.shape_b is not None:
        entry["shape_b"] = list(td.shape_b)
    for stat in ("mean_a", "mean_b", "max_abs_diff", "mean_abs_diff"):
        val = getattr(td, stat, None)
        if val is not None:
            entry[stat] = round(float(val), 8)
    return entry


def export_json(diff: CheckpointDiff, indent: int = 2) -> str:
    """Serialize a CheckpointDiff to a JSON string."""
    records = [_tensor_diff_to_dict(k, v) for k, v in diff.items()]
    return json.dumps(records, indent=indent)


def export_csv(diff: CheckpointDiff) -> str:
    """Serialize a CheckpointDiff to a CSV string."""
    fieldnames = [
        "key", "status", "shape_a", "shape_b",
        "mean_a", "mean_b", "max_abs_diff", "mean_abs_diff",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for key, td in diff.items():
        row = _tensor_diff_to_dict(key, td)
        row.setdefault("shape_a", "")
        row.setdefault("shape_b", "")
        writer.writerow(row)
    return output.getvalue()


def export_diff(diff: CheckpointDiff, fmt: str) -> str:
    """Export diff in the requested format ('json' or 'csv')."""
    fmt = fmt.lower()
    if fmt == "json":
        return export_json(diff)
    if fmt == "csv":
        return export_csv(diff)
    raise ValueError(f"Unsupported export format: {fmt!r}. Choose 'json' or 'csv'.")
