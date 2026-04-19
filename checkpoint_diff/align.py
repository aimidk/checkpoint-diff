"""Key alignment utilities for comparing checkpoints with renamed or prefixed keys."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np


def strip_prefix(keys: List[str], prefix: str) -> Dict[str, str]:
    """Return mapping from stripped key -> original key."""
    result = {}
    for k in keys:
        stripped = k[len(prefix):] if k.startswith(prefix) else k
        result[stripped] = k
    return result


def auto_detect_prefix(keys_a: List[str], keys_b: List[str]) -> Tuple[str, str]:
    """Heuristically detect differing prefixes between two key sets."""
    def common_prefix(keys: List[str]) -> str:
        if not keys:
            return ""
        prefix = keys[0]
        for k in keys[1:]:
            while not k.startswith(prefix):
                prefix = prefix[:-1]
                if not prefix:
                    return ""
        return prefix

    prefix_a = common_prefix(sorted(keys_a))
    prefix_b = common_prefix(sorted(keys_b))
    return prefix_a, prefix_b


def align_checkpoints(
    ckpt_a: Dict[str, np.ndarray],
    ckpt_b: Dict[str, np.ndarray],
    prefix_a: str = "",
    prefix_b: str = "",
    auto_align: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Return re-keyed copies of ckpt_a and ckpt_b with prefixes stripped."""
    if auto_align:
        prefix_a, prefix_b = auto_detect_prefix(list(ckpt_a.keys()), list(ckpt_b.keys()))

    def strip(ckpt: Dict[str, np.ndarray], prefix: str) -> Dict[str, np.ndarray]:
        if not prefix:
            return dict(ckpt)
        return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in ckpt.items()}

    return strip(ckpt_a, prefix_a), strip(ckpt_b, prefix_b)
