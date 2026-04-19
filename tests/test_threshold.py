"""Tests for checkpoint_diff.threshold."""
import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff, CheckpointDiff
from checkpoint_diff.threshold import (
    ThresholdConfig,
    FlaggedTensor,
    flag_tensors,
    format_flagged,
)


def _td(mean_a, mean_b, std_a=0.1, std_b=0.1, max_a=1.0, max_b=1.0):
    return TensorDiff(
        status="changed",
        shape_a=(4,),
        shape_b=(4,),
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        max_a=max_a,
        max_b=max_b,
    )


def _make_diff(**changed):
    return CheckpointDiff(changed=changed, added={}, removed={}, unchanged={})


def test_no_flags_when_within_threshold():
    diff = _make_diff(layer=_td(0.0, 0.05))
    cfg = ThresholdConfig(max_mean_delta=0.1)
    assert flag_tensors(diff, cfg) == []


def test_flags_mean_delta_exceeded():
    diff = _make_diff(layer=_td(0.0, 0.5))
    cfg = ThresholdConfig(max_mean_delta=0.1)
    result = flag_tensors(diff, cfg)
    assert len(result) == 1
    assert result[0].key == "layer"
    assert result[0].reason == "mean_delta"


def test_flags_std_delta_exceeded():
    diff = _make_diff(layer=_td(0.0, 0.0, std_a=0.1, std_b=0.9))
    cfg = ThresholdConfig(max_std_delta=0.5)
    result = flag_tensors(diff, cfg)
    assert len(result) == 1
    assert result[0].reason == "std_delta"


def test_flags_max_delta_exceeded():
    diff = _make_diff(layer=_td(0.0, 0.0, max_a=1.0, max_b=5.0))
    cfg = ThresholdConfig(max_max_delta=2.0)
    result = flag_tensors(diff, cfg)
    assert len(result) == 1
    assert result[0].reason == "max_delta"


def test_multiple_keys_flagged():
    diff = _make_diff(
        a=_td(0.0, 1.0),
        b=_td(0.0, 2.0),
    )
    cfg = ThresholdConfig(max_mean_delta=0.5)
    result = flag_tensors(diff, cfg)
    assert len(result) == 2


def test_format_flagged_no_flags():
    out = format_flagged([])
    assert "No tensors" in out


def test_format_flagged_with_flags():
    flags = [FlaggedTensor("w", "mean_delta", 0.42, 0.1)]
    out = format_flagged(flags)
    assert "w" in out
    assert "mean_delta" in out
