"""Slice utilities for inspecting sub-tensors within a checkpoint diff.

Provides helpers to extract and display a specific index range (slice)
of tensor data from a CheckpointDiff, useful for debugging large weight
matrices where only a portion of the values are of interest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class TensorSlice:
    """Holds a named slice of tensor data from both checkpoints."""

    key: str
    index: tuple  # numpy-compatible index expression
    values_a: Optional[np.ndarray]  # slice from checkpoint A (None if key absent)
    values_b: Optional[np.ndarray]  # slice from checkpoint B (None if key absent)
    delta: Optional[np.ndarray]     # element-wise difference (None if shapes differ)


def _parse_slice_spec(spec: str) -> tuple:
    """Parse a slice specification string into a tuple of slice objects.

    Supports comma-separated dimension specs such as "0:4", "1", or ":".

    Examples
    --------
    >>> _parse_slice_spec("0:4")
    (slice(0, 4, None),)
    >>> _parse_slice_spec("0:4,1:3")
    (slice(0, 4, None), slice(1, 3, None))
    >>> _parse_slice_spec("2")
    (2,)
    """
    parts = [p.strip() for p in spec.split(",")]
    indices: list = []
    for part in parts:
        if ":" in part:
            components = part.split(":")
            start = int(components[0]) if components[0] else None
            stop = int(components[1]) if len(components) > 1 and components[1] else None
            step = int(components[2]) if len(components) > 2 and components[2] else None
            indices.append(slice(start, stop, step))
        else:
            indices.append(int(part))
    return tuple(indices)


def slice_tensor_diff(diff: CheckpointDiff, key: str, spec: str) -> TensorSlice:
    """Extract a slice of tensor values for *key* from a CheckpointDiff.

    Parameters
    ----------
    diff:
        The full checkpoint diff to inspect.
    key:
        The weight key to slice.
    spec:
        A slice specification string, e.g. ``"0:4"`` or ``"0:2,0:3"``.

    Returns
    -------
    TensorSlice
        Object containing sliced arrays from A, B, and their delta.

    Raises
    ------
    KeyError
        If *key* is not present in the diff.
    """
    if key not in diff:
        raise KeyError(f"Key {key!r} not found in diff.")

    td: TensorDiff = diff[key]
    index = _parse_slice_spec(spec)

    values_a: Optional[np.ndarray] = None
    values_b: Optional[np.ndarray] = None
    delta: Optional[np.ndarray] = None

    if td.array_a is not None:
        values_a = td.array_a[index]
    if td.array_b is not None:
        values_b = td.array_b[index]
    if values_a is not None and values_b is not None:
        if values_a.shape == values_b.shape:
            delta = values_b.astype(float) - values_a.astype(float)

    return TensorSlice(key=key, index=index, values_a=values_a, values_b=values_b, delta=delta)


def format_slice(ts: TensorSlice, precision: int = 4) -> str:
    """Render a TensorSlice as a human-readable string.

    Parameters
    ----------
    ts:
        The TensorSlice to format.
    precision:
        Number of decimal places for floating-point values.
    """
    np.set_printoptions(precision=precision, suppress=True)
    lines: list[str] = [f"Key : {ts.key}", f"Index: {ts.index}"]

    if ts.values_a is not None:
        lines.append(f"A   :\n{ts.values_a}")
    else:
        lines.append("A   : <absent>")

    if ts.values_b is not None:
        lines.append(f"B   :\n{ts.values_b}")
    else:
        lines.append("B   : <absent>")

    if ts.delta is not None:
        lines.append(f"Δ   :\n{ts.delta}")
    elif ts.values_a is not None and ts.values_b is not None:
        lines.append("Δ   : <shape mismatch — cannot compute delta>")

    return "\n".join(lines)
