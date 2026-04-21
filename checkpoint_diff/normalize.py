"""Normalize checkpoint key names for comparison.

Provides utilities to strip common suffixes/prefixes, convert naming
conventions (e.g. camelCase -> snake_case), and apply custom regex
substitutions before diffing.
"""
from __future__ import annotations

import re
from typing import Dict, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Individual normalizers
# ---------------------------------------------------------------------------

def strip_suffix(keys: Dict[str, np.ndarray], suffix: str) -> Dict[str, np.ndarray]:
    """Remove *suffix* from every key that ends with it."""
    out: Dict[str, np.ndarray] = {}
    for k, v in keys.items():
        out[k[: -len(suffix)] if k.endswith(suffix) else k] = v
    return out


def camel_to_snake(keys: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert camelCase key segments to snake_case."""
    _pattern = re.compile(r'(?<=[a-z0-9])(?=[A-Z])')
    return {_pattern.sub('_', k).lower(): v for k, v in keys.items()}


def apply_regex(keys: Dict[str, np.ndarray], pattern: str, replacement: str) -> Dict[str, np.ndarray]:
    """Apply a regex substitution to every key name."""
    compiled = re.compile(pattern)
    return {compiled.sub(replacement, k): v for k, v in keys.items()}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

Normalizer = Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]


class NormalizePipeline:
    """Ordered sequence of normalizers applied left-to-right."""

    def __init__(self) -> None:
        self._steps: list[tuple[str, Normalizer]] = []

    def add(self, name: str, fn: Normalizer) -> "NormalizePipeline":
        self._steps.append((name, fn))
        return self

    def run(self, checkpoint: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        result = dict(checkpoint)
        for _name, fn in self._steps:
            result = fn(result)
        return result

    @property
    def step_names(self) -> list[str]:
        return [name for name, _ in self._steps]


def build_pipeline(
    *,
    camel_case: bool = False,
    strip_suffix_str: str | None = None,
    regex_sub: tuple[str, str] | None = None,
) -> NormalizePipeline:
    """Convenience factory to build a pipeline from common options."""
    pipeline = NormalizePipeline()
    if camel_case:
        pipeline.add("camel_to_snake", camel_to_snake)
    if strip_suffix_str:
        pipeline.add(
            f"strip_suffix:{strip_suffix_str}",
            lambda ck, s=strip_suffix_str: strip_suffix(ck, s),
        )
    if regex_sub:
        pattern, replacement = regex_sub
        pipeline.add(
            f"regex:{pattern}->{replacement}",
            lambda ck, p=pattern, r=replacement: apply_regex(ck, p, r),
        )
    return pipeline
