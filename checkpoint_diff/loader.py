"""Utilities for loading ML model checkpoints from various formats."""

import os
from pathlib import Path
from typing import Dict, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


SUPPORTED_EXTENSIONS = {".pt", ".pth", ".ckpt", ".npz", ".npy"}


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load a checkpoint file and return a flat dict of parameter tensors/arrays.

    Args:
        path: Path to the checkpoint file.

    Returns:
        Dictionary mapping parameter names to numpy arrays.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If the file extension is not supported.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ext = p.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported checkpoint format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext in {".pt", ".pth", ".ckpt"}:
        return _load_torch(path)
    elif ext == ".npz":
        return _load_npz(path)
    elif ext == ".npy":
        return _load_npy(path)


def _load_torch(path: str) -> Dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to load .pt/.pth/.ckpt files. Install it with: pip install torch")
    raw = torch.load(path, map_location="cpu")
    # Support state_dict directly or wrapped in a dict
    if isinstance(raw, dict):
        state = raw.get("state_dict", raw)
    else:
        state = raw.state_dict() if hasattr(raw, "state_dict") else {}
    return {k: v.numpy() for k, v in state.items() if hasattr(v, "numpy")}


def _load_npz(path: str) -> Dict[str, Any]:
    if not NUMPY_AVAILABLE:
        raise RuntimeError("NumPy is required to load .npz files. Install it with: pip install numpy")
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _load_npy(path: str) -> Dict[str, Any]:
    if not NUMPY_AVAILABLE:
        raise RuntimeError("NumPy is required to load .npy files. Install it with: pip install numpy")
    arr = np.load(path, allow_pickle=False)
    stem = Path(path).stem
    return {stem: arr}
