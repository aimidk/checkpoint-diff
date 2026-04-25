"""Sparsity analysis: compute zero-fraction and near-zero statistics for tensors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class SparsityRow:
    key: str
    status: str
    sparsity_a: Optional[float]  # fraction of exact zeros in checkpoint A
    sparsity_b: Optional[float]  # fraction of exact zeros in checkpoint B
    near_zero_a: Optional[float]  # fraction |x| < eps in A
    near_zero_b: Optional[float]  # fraction |x| < eps in B
    delta_sparsity: Optional[float]  # sparsity_b - sparsity_a


def _sparsity(arr: np.ndarray) -> float:
    return float(np.sum(arr == 0)) / arr.size if arr.size > 0 else 0.0


def _near_zero(arr: np.ndarray, eps: float) -> float:
    return float(np.sum(np.abs(arr) < eps)) / arr.size if arr.size > 0 else 0.0


def compute_sparsity(
    diff: CheckpointDiff,
    eps: float = 1e-6,
    include_unchanged: bool = False,
) -> List[SparsityRow]:
    rows: List[SparsityRow] = []
    for key, td in diff.items():
        if td.status == "unchanged" and not include_unchanged:
            continue
        a = td.tensor_a
        b = td.tensor_b
        sp_a = _sparsity(a) if a is not None else None
        sp_b = _sparsity(b) if b is not None else None
        nz_a = _near_zero(a, eps) if a is not None else None
        nz_b = _near_zero(b, eps) if b is not None else None
        delta = (sp_b - sp_a) if (sp_a is not None and sp_b is not None) else None
        rows.append(
            SparsityRow(
                key=key,
                status=td.status,
                sparsity_a=sp_a,
                sparsity_b=sp_b,
                near_zero_a=nz_a,
                near_zero_b=nz_b,
                delta_sparsity=delta,
            )
        )
    rows.sort(key=lambda r: abs(r.delta_sparsity) if r.delta_sparsity is not None else 0.0, reverse=True)
    return rows


def format_sparsity(rows: List[SparsityRow], top_n: Optional[int] = None) -> str:
    if not rows:
        return "No sparsity data available."
    display = rows[:top_n] if top_n else rows
    header = f"{'Key':<40} {'Status':<10} {'Sparse_A':>9} {'Sparse_B':>9} {'NearZero_A':>11} {'NearZero_B':>11} {'Delta':>8}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in display:
        def _fmt(v: Optional[float]) -> str:
            return f"{v:.4f}" if v is not None else "  N/A  "
        lines.append(
            f"{r.key:<40} {r.status:<10} {_fmt(r.sparsity_a):>9} {_fmt(r.sparsity_b):>9}"
            f" {_fmt(r.near_zero_a):>11} {_fmt(r.near_zero_b):>11} {_fmt(r.delta_sparsity):>8}"
        )
    return "\n".join(lines)
