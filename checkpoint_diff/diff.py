"""Compute diffs between two model checkpoints."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class TensorDiff:
    key: str
    status: str  # 'added', 'removed', 'changed', 'unchanged'
    shape_a: Optional[tuple] = None
    shape_b: Optional[tuple] = None
    max_abs_diff: Optional[float] = None
    mean_abs_diff: Optional[float] = None


@dataclass
class CheckpointDiff:
    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    changed: List[TensorDiff] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)

    @property
    def has_differences(self) -> bool:
        return bool(self.added or self.removed or self.changed)


def compute_diff(
    ckpt_a: Dict[str, np.ndarray],
    ckpt_b: Dict[str, np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> CheckpointDiff:
    """Compare two checkpoints and return a structured diff."""
    keys_a = set(ckpt_a.keys())
    keys_b = set(ckpt_b.keys())

    result = CheckpointDiff(
        added=sorted(keys_b - keys_a),
        removed=sorted(keys_a - keys_b),
    )

    for key in sorted(keys_a & keys_b):
        arr_a = np.asarray(ckpt_a[key], dtype=float)
        arr_b = np.asarray(ckpt_b[key], dtype=float)

        if arr_a.shape != arr_b.shape:
            result.changed.append(
                TensorDiff(
                    key=key,
                    status="changed",
                    shape_a=arr_a.shape,
                    shape_b=arr_b.shape,
                )
            )
        elif np.allclose(arr_a, arr_b, rtol=rtol, atol=atol):
            result.unchanged.append(key)
        else:
            diff = np.abs(arr_a - arr_b)
            result.changed.append(
                TensorDiff(
                    key=key,
                    status="changed",
                    shape_a=arr_a.shape,
                    shape_b=arr_b.shape,
                    max_abs_diff=float(diff.max()),
                    mean_abs_diff=float(diff.mean()),
                )
            )

    return result
