"""Format and print checkpoint diffs as human-readable reports."""

from typing import TextIO
import sys

from checkpoint_diff.diff import CheckpointDiff, TensorDiff


def _fmt_shape(shape) -> str:
    return "(" + ", ".join(str(d) for d in shape) + ")"


def format_report(diff: CheckpointDiff, verbose: bool = False) -> str:
    lines = []

    if not diff.has_differences:
        lines.append("Checkpoints are identical.")
        return "\n".join(lines)

    if diff.added:
        lines.append(f"Added keys ({len(diff.added)}):")
        for key in diff.added:
            lines.append(f"  + {key}")

    if diff.removed:
        lines.append(f"Removed keys ({len(diff.removed)}):")
        for key in diff.removed:
            lines.append(f"  - {key}")

    if diff.changed:
        lines.append(f"Changed tensors ({len(diff.changed)}):")
        for td in diff.changed:
            shape_info = f"{_fmt_shape(td.shape_a)} -> {_fmt_shape(td.shape_b)}"
            if td.max_abs_diff is not None:
                lines.append(
                    f"  ~ {td.key}  shape={shape_info}"
                    f"  max_diff={td.max_abs_diff:.4e}"
                    f"  mean_diff={td.mean_abs_diff:.4e}"
                )
            else:
                lines.append(f"  ~ {td.key}  shape={shape_info}")

    if verbose and diff.unchanged:
        lines.append(f"Unchanged tensors ({len(diff.unchanged)}):")
        for key in diff.unchanged:
            lines.append(f"    {key}")

    return "\n".join(lines)


def print_report(
    diff: CheckpointDiff,
    verbose: bool = False,
    file: TextIO = sys.stdout,
) -> None:
    print(format_report(diff, verbose=verbose), file=file)
