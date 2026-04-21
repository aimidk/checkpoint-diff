"""Key rename mapping: apply a rename map to align checkpoint keys before diffing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

RenameMap = Dict[str, str]


def load_rename_map(path: str | Path) -> RenameMap:
    """Load a JSON file mapping old key names to new key names."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Rename map must be a JSON object, got {type(data).__name__}")
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(f"All keys and values must be strings; got {k!r}: {v!r}")
    return data


def apply_rename_map(
    checkpoint: dict[str, np.ndarray],
    rename_map: RenameMap,
    *,
    strict: bool = False,
) -> dict[str, np.ndarray]:
    """Return a new checkpoint with keys renamed according to *rename_map*.

    Parameters
    ----------
    checkpoint:
        Source checkpoint dictionary.
    rename_map:
        Mapping of ``{old_key: new_key}``.
    strict:
        If ``True``, raise ``KeyError`` when a key listed in *rename_map* is
        absent from *checkpoint*.  Defaults to ``False`` (silently skip).
    """
    result: dict[str, np.ndarray] = {}
    for key, value in checkpoint.items():
        new_key = rename_map.get(key, key)
        result[new_key] = value

    if strict:
        missing = set(rename_map) - set(checkpoint)
        if missing:
            raise KeyError(f"Keys in rename map not found in checkpoint: {sorted(missing)}")

    return result


def invert_rename_map(rename_map: RenameMap) -> RenameMap:
    """Return the inverse mapping ``{new_key: old_key}``.

    Raises ``ValueError`` if the map is not bijective.
    """
    inverted: RenameMap = {}
    for old, new in rename_map.items():
        if new in inverted:
            raise ValueError(
                f"Rename map is not bijective: '{new}' maps to both "
                f"'{inverted[new]}' and '{old}'"
            )
        inverted[new] = old
    return inverted
