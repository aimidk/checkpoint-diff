"""Tests for checkpoint_diff.normalize."""
from __future__ import annotations

import numpy as np
import pytest

from checkpoint_diff.normalize import (
    strip_suffix,
    camel_to_snake,
    apply_regex,
    NormalizePipeline,
    build_pipeline,
)


def _ck(**kwargs: list) -> dict:
    return {k: np.array(v) for k, v in kwargs.items()}


# ---------------------------------------------------------------------------
# strip_suffix
# ---------------------------------------------------------------------------

def test_strip_suffix_removes_matching():
    ck = _ck(**{"weight.data": [1.0], "bias.data": [2.0]})
    result = strip_suffix(ck, ".data")
    assert set(result.keys()) == {"weight", "bias"}


def test_strip_suffix_leaves_non_matching_unchanged():
    ck = _ck(**{"weight.data": [1.0], "other": [2.0]})
    result = strip_suffix(ck, ".data")
    assert "other" in result
    assert "weight" in result


# ---------------------------------------------------------------------------
# camel_to_snake
# ---------------------------------------------------------------------------

def test_camel_to_snake_converts_keys():
    ck = _ck(**{"layerNorm": [1.0], "hiddenState": [2.0]})
    result = camel_to_snake(ck)
    assert "layer_norm" in result
    assert "hidden_state" in result


def test_camel_to_snake_leaves_snake_unchanged():
    ck = _ck(**{"layer_norm": [1.0]})
    result = camel_to_snake(ck)
    assert "layer_norm" in result


# ---------------------------------------------------------------------------
# apply_regex
# ---------------------------------------------------------------------------

def test_apply_regex_substitutes_pattern():
    ck = _ck(**{"module.layer.0.weight": [1.0], "module.layer.1.weight": [2.0]})
    result = apply_regex(ck, r"module\.", "")
    assert "layer.0.weight" in result
    assert "layer.1.weight" in result


def test_apply_regex_no_match_leaves_unchanged():
    ck = _ck(**{"weight": [1.0]})
    result = apply_regex(ck, r"bias", "b")
    assert "weight" in result


# ---------------------------------------------------------------------------
# NormalizePipeline
# ---------------------------------------------------------------------------

def test_pipeline_applies_steps_in_order():
    ck = _ck(**{"module.layerNorm.data": [1.0]})
    pipeline = (
        NormalizePipeline()
        .add("camel", camel_to_snake)
        .add("suffix", lambda c: strip_suffix(c, ".data"))
    )
    result = pipeline.run(ck)
    assert "module.layer_norm" in result


def test_pipeline_step_names_recorded():
    pipeline = NormalizePipeline()
    pipeline.add("step_a", lambda c: c)
    pipeline.add("step_b", lambda c: c)
    assert pipeline.step_names == ["step_a", "step_b"]


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------

def test_build_pipeline_camel_flag():
    ck = _ck(**{"hiddenSize": [1.0]})
    pipeline = build_pipeline(camel_case=True)
    result = pipeline.run(ck)
    assert "hidden_size" in result


def test_build_pipeline_empty_is_identity():
    ck = _ck(**{"weight": [1.0]})
    pipeline = build_pipeline()
    result = pipeline.run(ck)
    assert "weight" in result


def test_build_pipeline_regex_sub():
    ck = _ck(**{"encoder.weight": [1.0]})
    pipeline = build_pipeline(regex_sub=(r"encoder\.", "enc."))
    result = pipeline.run(ck)
    assert "enc.weight" in result
