"""Tests for checkpoint_diff.cluster and checkpoint_diff.cli_cluster."""
from __future__ import annotations

import argparse
import io
from typing import Dict

import numpy as np
import pytest

from checkpoint_diff.cluster import (
    ClusterResult,
    cluster_by_magnitude,
    format_clusters,
)
from checkpoint_diff.cli_cluster import add_cluster_args, apply_clustering
from checkpoint_diff.diff import CheckpointDiff, TensorDiff


def _td(status="changed", mean_b=0.0, std_b=1.0) -> TensorDiff:
    a = np.array([0.0])
    b = np.array([mean_b])
    return TensorDiff(
        status=status,
        shape_a=a.shape,
        shape_b=b.shape,
        mean_a=float(a.mean()),
        mean_b=mean_b,
        std_a=float(a.std()),
        std_b=std_b,
        max_abs_diff=abs(mean_b),
    )


def _make_diff(tensors: Dict[str, TensorDiff]) -> CheckpointDiff:
    return CheckpointDiff(tensors=tensors)


# ---------------------------------------------------------------------------
# cluster_by_magnitude
# ---------------------------------------------------------------------------

def test_empty_diff_returns_empty():
    diff = _make_diff({})
    assert cluster_by_magnitude(diff) == []


def test_returns_n_bins_clusters():
    diff = _make_diff({
        "a": _td(mean_b=0.1),
        "b": _td(mean_b=1.0),
        "c": _td(mean_b=5.0),
    })
    clusters = cluster_by_magnitude(diff, n_bins=3)
    assert len(clusters) == 3


def test_all_keys_distributed():
    diff = _make_diff({
        "a": _td(mean_b=0.1),
        "b": _td(mean_b=0.5),
        "c": _td(mean_b=0.9),
    })
    clusters = cluster_by_magnitude(diff, n_bins=3)
    total_keys = sum(c.size for c in clusters)
    assert total_keys == 3


def test_removed_keys_excluded_by_default():
    diff = _make_diff({
        "x": _td(status="removed", mean_b=99.0),
        "y": _td(status="changed", mean_b=1.0),
    })
    clusters = cluster_by_magnitude(diff, n_bins=2)
    total = sum(c.size for c in clusters)
    assert total == 1


def test_custom_status_filter():
    diff = _make_diff({
        "x": _td(status="added", mean_b=1.0),
        "y": _td(status="changed", mean_b=2.0),
    })
    clusters = cluster_by_magnitude(diff, n_bins=2, statuses=["added"])
    total = sum(c.size for c in clusters)
    assert total == 1


def test_cluster_result_centroid_mean_reasonable():
    diff = _make_diff({"a": _td(mean_b=2.0, std_b=0.5)})
    clusters = cluster_by_magnitude(diff, n_bins=1)
    non_empty = [c for c in clusters if c.size > 0]
    assert len(non_empty) == 1
    assert pytest.approx(non_empty[0].centroid_mean, abs=1e-6) == 2.0


# ---------------------------------------------------------------------------
# format_clusters
# ---------------------------------------------------------------------------

def test_format_clusters_empty():
    result = format_clusters([])
    assert "No clusters" in result


def test_format_clusters_contains_band_label():
    clusters = [ClusterResult(label="band_0: [0, 1)", keys=["w"], centroid_mean=0.5, centroid_std=0.1)]
    out = format_clusters(clusters)
    assert "band_0" in out
    assert "w" not in out  # keys are not listed individually


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_cluster_args(p)
    return p


def test_add_cluster_args_registers_flag():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.cluster is False


def test_add_cluster_args_bins_default():
    p = _make_parser()
    ns = p.parse_args([])
    assert ns.cluster_bins == 3


def test_apply_clustering_not_triggered_without_flag():
    p = _make_parser()
    ns = p.parse_args([])
    diff = _make_diff({"a": _td()})
    result = apply_clustering(ns, diff)
    assert result is None


def test_apply_clustering_writes_to_out():
    p = _make_parser()
    ns = p.parse_args(["--cluster", "--cluster-bins", "2"])
    diff = _make_diff({"a": _td(mean_b=0.5), "b": _td(mean_b=2.0)})
    buf = io.StringIO()
    result = apply_clustering(ns, diff, out=buf)
    assert result is not None
    assert "band_" in buf.getvalue()
