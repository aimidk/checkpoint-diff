"""compare_sets: compare multiple checkpoints against a reference."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from checkpoint_diff.diff import CheckpointDiff, compute_diff
from checkpoint_diff.diff_score import DiffScore, compute_score


@dataclass
class CompareEntry:
    label: str
    diff: CheckpointDiff
    score: DiffScore


@dataclass
class CompareSetResult:
    reference_label: str
    entries: List[CompareEntry] = field(default_factory=list)

    def ranked(self) -> List[CompareEntry]:
        """Return entries sorted by overall diff score descending."""
        return sorted(self.entries, key=lambda e: e.score.overall, reverse=True)


def compare_against_reference(
    reference: Dict[str, object],
    candidates: Dict[str, Dict[str, object]],
    reference_label: str = "reference",
) -> CompareSetResult:
    """Diff each candidate checkpoint against *reference*.

    Args:
        reference: Mapping of tensor name -> array for the reference checkpoint.
        candidates: Mapping of label -> checkpoint dict for each candidate.
        reference_label: Human-readable name for the reference checkpoint.

    Returns:
        A :class:`CompareSetResult` containing one entry per candidate.
    """
    result = CompareSetResult(reference_label=reference_label)
    for label, ckpt in candidates.items():
        diff = compute_diff(reference, ckpt)
        score = compute_score(diff)
        result.entries.append(CompareEntry(label=label, diff=diff, score=score))
    return result


def format_compare_set(result: CompareSetResult, top_n: Optional[int] = None) -> str:
    """Render a ranked comparison table as a string."""
    ranked = result.ranked()
    if top_n is not None:
        ranked = ranked[:top_n]

    header = f"Comparison against reference: {result.reference_label}"
    separator = "-" * 60
    lines = [header, separator]
    col_w = max((len(e.label) for e in ranked), default=10)
    lines.append(f"  {'Label':<{col_w}}  {'Score':>8}  {'Changed':>8}  {'Added':>6}  {'Removed':>8}")
    lines.append(separator)
    for entry in ranked:
        s = entry.score
        lines.append(
            f"  {entry.label:<{col_w}}  {s.overall:>8.4f}"
            f"  {s.num_changed:>8}  {s.num_added:>6}  {s.num_removed:>8}"
        )
    lines.append(separator)
    return "\n".join(lines)
