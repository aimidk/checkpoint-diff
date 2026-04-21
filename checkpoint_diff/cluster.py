"""Group tensor diffs into clusters based on statistical similarity."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


@dataclass
class ClusterResult:
    label: str
    keys: List[str] = field(default_factory=list)
    centroid_mean: float = 0.0
    centroid_std: float = 0.0

    @property
    def size(self) -> int:
        return len(self.keys)


def _feature_vector(td: TensorDiff) -> Optional[np.ndarray]:
    """Return a 2-element feature vector [mean_b, std_b] or None if unavailable."""
    if td.mean_b is None or td.std_b is None:
        return None
    return np.array([td.mean_b, td.std_b], dtype=float)


def cluster_by_magnitude(
    diff: CheckpointDiff,
    n_bins: int = 3,
    statuses: Optional[List[str]] = None,
) -> List[ClusterResult]:
    """Cluster changed/added tensors into *n_bins* magnitude bands by |mean_b|.

    Parameters
    ----------
    diff:     CheckpointDiff to analyse.
    n_bins:   Number of equal-width bins in log-space (default 3).
    statuses: Limit to these status strings; defaults to ["changed", "added"].
    """
    if statuses is None:
        statuses = ["changed", "added"]

    candidates: Dict[str, TensorDiff] = {
        k: v for k, v in diff.tensors.items() if v.status in statuses
    }

    if not candidates:
        return []

    vectors = {k: _feature_vector(v) for k, v in candidates.items()}
    valid = {k: v for k, v in vectors.items() if v is not None}

    if not valid:
        return []

    abs_means = np.array([abs(v[0]) for v in valid.values()])
    keys_ordered = list(valid.keys())

    # Use linear bins if all values are zero
    mn, mx = float(abs_means.min()), float(abs_means.max())
    if mx == 0.0:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.linspace(mn, mx + 1e-12, n_bins + 1)

    clusters: List[ClusterResult] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        label = f"band_{i}: [{lo:.3g}, {hi:.3g})"
        result = ClusterResult(label=label)
        for idx, key in enumerate(keys_ordered):
            val = abs_means[idx]
            if lo <= val < hi:
                result.keys.append(key)
                result.centroid_mean += valid[key][0]
                result.centroid_std += valid[key][1]
        if result.size:
            result.centroid_mean /= result.size
            result.centroid_std /= result.size
        clusters.append(result)

    return clusters


def format_clusters(clusters: List[ClusterResult]) -> str:
    """Return a human-readable table of cluster results."""
    if not clusters:
        return "No clusters to display."
    lines = [f"{'Band':<30} {'Keys':>6} {'Centroid Mean':>14} {'Centroid Std':>13}"]
    lines.append("-" * 67)
    for c in clusters:
        lines.append(
            f"{c.label:<30} {c.size:>6} {c.centroid_mean:>14.4f} {c.centroid_std:>13.4f}"
        )
    return "\n".join(lines)
