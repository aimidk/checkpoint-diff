"""Tests for checkpoint_diff.summary."""
import pytest
from checkpoint_diff.diff import TensorDiff, CheckpointDiff
from checkpoint_diff.summary import summarize, format_summary, DiffSummary


def _make_diff(**entries) -> CheckpointDiff:
    return CheckpointDiff(entries)


def _td(status, mean_delta=None, shape_a=None, shape_b=None):
    return TensorDiff(
        status=status,
        shape_a=shape_a or (4,),
        shape_b=shape_b or (4,),
        mean_a=0.0,
        mean_b=0.0 if mean_delta is None else mean_delta,
        std_a=1.0,
        std_b=1.0,
        mean_delta=mean_delta,
        max_abs_delta=None,
    )


def test_summary_counts():
    diff = _make_diff(
        a=_td("added"),
        b=_td("removed"),
        c=_td("changed", mean_delta=0.5),
        d=_td("unchanged"),
    )
    s = summarize(diff)
    assert s.total_keys == 4
    assert s.added == 1
    assert s.removed == 1
    assert s.changed == 1
    assert s.unchanged == 1


def test_summary_most_changed_key():
    diff = _make_diff(
        layer1=_td("changed", mean_delta=0.1),
        layer2=_td("changed", mean_delta=-3.5),
        layer3=_td("changed", mean_delta=2.0),
    )
    s = summarize(diff)
    assert s.most_changed_key == "layer2"
    assert abs(s.max_abs_mean_change - 3.5) < 1e-9


def test_summary_no_changed_keys():
    diff = _make_diff(a=_td("unchanged"), b=_td("added"))
    s = summarize(diff)
    assert s.most_changed_key is None
    assert s.max_abs_mean_change is None


def test_summary_as_dict_keys():
    diff = _make_diff(x=_td("unchanged"))
    d = summarize(diff).as_dict()
    for key in ("total_keys", "added", "removed", "changed", "unchanged",
                "max_abs_mean_change", "most_changed_key"):
        assert key in d


def test_format_summary_contains_counts():
    diff = _make_diff(
        a=_td("added"),
        b=_td("changed", mean_delta=1.23),
    )
    s = summarize(diff)
    text = format_summary(s)
    assert "Added" in text
    assert "Changed" in text
    assert "most changed key" in text.lower() or "Most changed" in text


def test_format_summary_no_most_changed():
    diff = _make_diff(a=_td("unchanged"))
    text = format_summary(summarize(diff))
    assert "most changed" not in text.lower()
