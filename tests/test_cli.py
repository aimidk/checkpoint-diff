"""Tests for the CLI entry point."""

import numpy as np
import pytest

from checkpoint_diff.cli import main


@pytest.fixture()
def identical_npz(tmp_path):
    data = {"weight": np.array([1.0, 2.0, 3.0])}
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    np.savez(str(a), **data)
    np.savez(str(b), **data)
    return str(a), str(b)


@pytest.fixture()
def different_npz(tmp_path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    np.savez(str(a), weight=np.array([1.0, 2.0, 3.0]))
    np.savez(str(b), weight=np.array([4.0, 5.0, 6.0]))
    return str(a), str(b)


def test_identical_exits_zero(identical_npz):
    a, b = identical_npz
    assert main([a, b]) == 0


def test_identical_exits_zero_with_exit_code_flag(identical_npz):
    a, b = identical_npz
    assert main([a, b, "--exit-code"]) == 0


def test_different_exits_zero_without_flag(different_npz):
    a, b = different_npz
    assert main([a, b]) == 0


def test_different_exits_one_with_exit_code_flag(different_npz):
    a, b = different_npz
    assert main([a, b, "--exit-code"]) == 1


def test_missing_file_exits_two(tmp_path):
    real = tmp_path / "real.npz"
    np.savez(str(real), x=np.array([1.0]))
    assert main([str(real), "nonexistent.npz"]) == 2


def test_unsupported_extension_exits_two(tmp_path):
    f = tmp_path / "model.pkl"
    f.write_bytes(b"data")
    assert main([str(f), str(f)]) == 2


def test_keys_filter(tmp_path):
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    np.savez(str(a), weight=np.array([1.0]), bias=np.array([0.0]))
    np.savez(str(b), weight=np.array([2.0]), bias=np.array([0.0]))
    # filtering to unchanged key → no differences
    assert main([str(a), str(b), "--exit-code", "--keys", "bias"]) == 0
    # filtering to changed key → differences
    assert main([str(a), str(b), "--exit-code", "--keys", "weight"]) == 1


def test_no_color_flag_does_not_crash(identical_npz):
    a, b = identical_npz
    assert main([a, b, "--no-color"]) == 0
