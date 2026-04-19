"""Tests for checkpoint_diff.cli_similarity."""
import argparse
import os
import tempfile

import numpy as np
import pytest

from checkpoint_diff.diff import TensorDiff, CheckpointDiff
from checkpoint_diff.cli_similarity import add_similarity_args, apply_similarity


def _make_parser():
    p = argparse.ArgumentParser()
    add_similarity_args(p)
    return p


def _diff():
    td = TensorDiff(
        status="changed",
        array_a=np.array([1.0, 0.0]),
        array_b=np.array([0.8, 0.2]),
    )
    return CheckpointDiff({"w": td})


def test_add_similarity_args_registers_flag():
    p = _make_parser()
    args = p.parse_args([])
    assert hasattr(args, "similarity")
    assert args.similarity is False


def test_add_similarity_args_export_default_none():
    p = _make_parser()
    args = p.parse_args([])
    assert args.similarity_export is None


def test_similarity_flag_parsed():
    p = _make_parser()
    args = p.parse_args(["--similarity"])
    assert args.similarity is True


def test_apply_similarity_prints(capsys):
    p = _make_parser()
    args = p.parse_args(["--similarity"])
    apply_similarity(args, _diff())
    out = capsys.readouterr().out
    assert "Similarity Report" in out
    assert "w" in out


def test_apply_similarity_no_flag_no_output(capsys):
    p = _make_parser()
    args = p.parse_args([])
    apply_similarity(args, _diff())
    out = capsys.readouterr().out
    assert out == ""


def test_apply_similarity_export_writes_file():
    p = _make_parser()
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        path = f.name
    try:
        args = p.parse_args(["--similarity-export", path])
        apply_similarity(args, _diff())
        content = open(path).read()
        assert "w" in content
    finally:
        os.unlink(path)
