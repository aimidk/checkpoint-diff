"""CLI integration for tensor slicing functionality.

Allows users to inspect specific slices of tensors in a diff,
e.g. --slice "layer.weight:0:10,0:5" to view a sub-region.
"""

from __future__ import annotations

import argparse
from typing import Optional

from checkpoint_diff.diff import CheckpointDiff
from checkpoint_diff.slice import format_slice, slice_tensor_diff


def add_slice_args(parser: argparse.ArgumentParser) -> None:
    """Register tensor-slicing arguments on *parser*.

    Arguments added
    ---------------
    --slice KEY:SPEC
        Slice a single tensor by key and numpy-style slice spec.
        Example: ``--slice encoder.weight:0:4,0:8``
    --slice-top N
        Print the first N rows of every changed tensor (shorthand).
    """
    group = parser.add_argument_group("tensor slicing")
    group.add_argument(
        "--slice",
        dest="slice_spec",
        metavar="KEY:SPEC",
        default=None,
        help=(
            "Slice a tensor by key and index spec. "
            "Format: 'key:dim0_start:dim0_end[,dim1_start:dim1_end,...]'. "
            "Example: --slice encoder.weight:0:4,0:8"
        ),
    )
    group.add_argument(
        "--slice-top",
        dest="slice_top",
        metavar="N",
        type=int,
        default=None,
        help="Show the first N rows of every changed tensor.",
    )


def _parse_key_spec(raw: str) -> tuple[str, str]:
    """Split ``KEY:SPEC`` into ``(key, spec)``.

    The key itself may contain dots but not colons, so the first colon
    is used as the delimiter.

    Raises
    ------
    ValueError
        If *raw* does not contain a colon separator.
    """
    if ":" not in raw:
        raise ValueError(
            f"--slice value must be in 'KEY:SPEC' format, got: {raw!r}"
        )
    key, _, spec = raw.partition(":")
    return key.strip(), spec.strip()


def apply_slicing(
    args: argparse.Namespace,
    diff: CheckpointDiff,
) -> Optional[str]:
    """Apply slice arguments from *args* to *diff* and return formatted output.

    Returns ``None`` when no slicing flags were supplied so callers can
    skip the section entirely.

    Parameters
    ----------
    args:
        Parsed CLI namespace; expected to carry ``slice_spec`` and
        ``slice_top`` attributes (added by :func:`add_slice_args`).
    diff:
        The :class:`~checkpoint_diff.diff.CheckpointDiff` to inspect.

    Returns
    -------
    str or None
        Formatted slice report, or ``None`` if no slice flags were set.
    """
    slice_spec: Optional[str] = getattr(args, "slice_spec", None)
    slice_top: Optional[int] = getattr(args, "slice_top", None)

    if slice_spec is None and slice_top is None:
        return None

    lines: list[str] = []

    if slice_spec is not None:
        try:
            key, spec = _parse_key_spec(slice_spec)
        except ValueError as exc:
            return f"[slice error] {exc}"

        result = slice_tensor_diff(diff, key, spec)
        if result is None:
            return f"[slice] key {key!r} not found or tensor unchanged."
        lines.append(format_slice(result))

    if slice_top is not None:
        # Build a synthetic "0:N" spec for every changed key.
        top_spec = f"0:{slice_top}"
        for key, td in diff.changed.items():
            result = slice_tensor_diff(diff, key, top_spec)
            if result is not None:
                lines.append(format_slice(result))

    return "\n".join(lines) if lines else None
