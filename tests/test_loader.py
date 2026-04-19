"""Tests for checkpoint_diff.loader module."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from checkpoint_diff.loader import load_checkpoint, SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_npz(tmp_path: Path, name: str = "ckpt.npz") -> Path:
    p = tmp_path / name
    np.savez(p, weight=np.array([1.0, 2.0]), bias=np.array([0.5]))
    return p


def make_npy(tmp_path: Path, name: str = "layer.npy") -> Path:
    p = tmp_path / name
    np.save(p, np.array([[1.0, 2.0], [3.0, 4.0]]))
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadCheckpoint:
    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_checkpoint(str(tmp_path / "missing.pt"))

    def test_unsupported_extension_raises(self, tmp_path):
        bad = tmp_path / "model.h5"
        bad.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported checkpoint format"):
            load_checkpoint(str(bad))

    def test_supported_extensions_set(self):
        assert ".pt" in SUPPORTED_EXTENSIONS
        assert ".npz" in SUPPORTED_EXTENSIONS
        assert ".npy" in SUPPORTED_EXTENSIONS

    def test_load_npz(self, tmp_path):
        p = make_npz(tmp_path)
        result = load_checkpoint(str(p))
        assert set(result.keys()) == {"weight", "bias"}
        np.testing.assert_array_equal(result["weight"], np.array([1.0, 2.0]))
        np.testing.assert_array_equal(result["bias"], np.array([0.5]))

    def test_load_npy(self, tmp_path):
        p = make_npy(tmp_path)
        result = load_checkpoint(str(p))
        assert "layer" in result
        assert result["layer"].shape == (2, 2)

    def test_load_torch_state_dict(self, tmp_path):
        """Mock torch loading so test runs without PyTorch installed."""
        fake_tensor = MagicMock()
        fake_tensor.numpy.return_value = np.array([1.0, 2.0])
        fake_state = {"fc.weight": fake_tensor}

        pt_path = tmp_path / "model.pt"
        pt_path.write_bytes(b"")

        with patch("checkpoint_diff.loader.TORCH_AVAILABLE", True), \
             patch("checkpoint_diff.loader.torch") as mock_torch:
            mock_torch.load.return_value = fake_state
            result = load_checkpoint(str(pt_path))

        assert "fc.weight" in result
        np.testing.assert_array_equal(result["fc.weight"], np.array([1.0, 2.0]))

    def test_load_torch_not_available(self, tmp_path):
        pt_path = tmp_path / "model.pt"
        pt_path.write_bytes(b"")
        with patch("checkpoint_diff.loader.TORCH_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="PyTorch is required"):
                load_checkpoint(str(pt_path))
