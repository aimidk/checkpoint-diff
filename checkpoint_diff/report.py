"""Formatting and printing of CheckpointDiff reports."""

from __future__ import annotations

from typing import TextIO
import sys

from checkpoint_diff.diff import CheckpointDiff, has_differences

_STATUS_LABEL = {
    "added": "[+]",
    "removed": "[-]",
    "changed": "[~]",
    "unchanged": "[=]",
}


def _fmt_shape(shape: tuple[int, ...] | None) -> str:
    if shape is None:
        return "n/a"
    return "(" + ", ".join(str(d) for d in shape) + ")"


def _fmt_stat(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.6f}"


def format_report(diff: CheckpointDiff, *, show_unchanged: bool = False) -> str:
    lines: list[str] = []
    total = len(diff)
    changed_count = sum(1 for td in diff.values() if td.status != "unchanged")
    lines.append(f"Checkpoint diff — {total} keys, {changed_count} with differences")
    lines.append("-" * 60)

    for key in sorted(diff):
        td = diff[key]
        if td.status == "unchanged" and not show_unchanged:
            continue
        label = _STATUS_LABEL.get(td.status, "[?]")
        shape_a = _fmt_shape(td.shape_a)
        shape_b = _fmt_shape(td.shape_b)
        line = f"{label} {key}  shape: {shape_a} -> {shape_b}"
        if td.status == "changed":
            line += (
                f"  mean_diff={_fmt_stat(td.mean_diff)}"
                f"  max_abs_diff={_fmt_stat(td.max_abs_diff)}"
            )
        lines.append(line)

    if not has_differences(diff):
        lines.append("Checkpoints are identical.")
    return "\n".join(lines)


def print_report(
    diff: CheckpointDiff,
    *,
    show_unchanged: bool = False,
    file: TextIO = sys.stdout,
) -> None:
    print(format_report(diff, show_unchanged=show_unchanged), file=file)
