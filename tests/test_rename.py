"""Tests for checkpoint_diff.rename."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

from checkpoint_diff.rename import (
    apply_rename_map,
    invert_rename_map,
    load_rename_map,
)


@pytest.fixture()
def rename_json(tmp_path: Path) -> Path:
    mapping = {"layer.0.weight": "encoder.0.weight", "layer.0.bias": "encoder.0.bias"}
    p = tmp_path / "rename.json"
    p.write_text(json.dumps(mapping))
    return p


# ---------------------------------------------------------------------------
# load_rename_map
# ---------------------------------------------------------------------------

def test_load_rename_map_returns_dict(rename_json: Path) -> None:
    result = load_rename_map(rename_json)
    assert result == {"layer.0.weight": "encoder.0.weight", "layer.0.bias": "encoder.0.bias"}


def test_load_rename_map_invalid_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="JSON object"):
        load_rename_map(p)


def test_load_rename_map_non_string_value_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"a": 1}))
    with pytest.raises(ValueError, match="strings"):
        load_rename_map(p)


# ---------------------------------------------------------------------------
# apply_rename_map
# ---------------------------------------------------------------------------

def _ckpt() -> dict[str, np.ndarray]:
    return {
        "layer.0.weight": np.ones((4, 4)),
        "layer.0.bias": np.zeros((4,)),
        "head.weight": np.eye(4),
    }


def test_apply_rename_map_renames_keys() -> None:
    mapping = {"layer.0.weight": "encoder.0.weight"}
    result = apply_rename_map(_ckpt(), mapping)
    assert "encoder.0.weight" in result
    assert "layer.0.weight" not in result


def test_apply_rename_map_preserves_unmapped_keys() -> None:
    mapping = {"layer.0.weight": "encoder.0.weight"}
    result = apply_rename_map(_ckpt(), mapping)
    assert "head.weight" in result
    assert "layer.0.bias" in result


def test_apply_rename_map_values_unchanged() -> None:
    mapping = {"layer.0.weight": "enc.w"}
    ckpt = _ckpt()
    result = apply_rename_map(ckpt, mapping)
    np.testing.assert_array_equal(result["enc.w"], ckpt["layer.0.weight"])


def test_apply_rename_map_strict_raises_on_missing_key() -> None:
    mapping = {"nonexistent": "something"}
    with pytest.raises(KeyError, match="nonexistent"):
        apply_rename_map(_ckpt(), mapping, strict=True)


def test_apply_rename_map_non_strict_ignores_missing_key() -> None:
    mapping = {"nonexistent": "something"}
    result = apply_rename_map(_ckpt(), mapping, strict=False)
    assert "something" not in result


# ---------------------------------------------------------------------------
# invert_rename_map
# ---------------------------------------------------------------------------

def test_invert_rename_map_reverses_mapping() -> None:
    mapping = {"a": "b", "c": "d"}
    assert invert_rename_map(mapping) == {"b": "a", "d": "c"}


def test_invert_rename_map_non_bijective_raises() -> None:
    mapping = {"a": "x", "b": "x"}
    with pytest.raises(ValueError, match="bijective"):
        invert_rename_map(mapping)
