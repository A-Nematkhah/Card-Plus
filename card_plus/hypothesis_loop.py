"""Phase-5: hypothesis ← confirm/reject memory across CARD iterations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PathLike = Union[str, Path]
MEMORY_NAME = "hypothesis_memory.jsonl"


@dataclass
class HypothesisRecord:
    iteration: int
    hypothesis: str
    expected_effect: str
    metric_before: Optional[float] = None
    metric_after: Optional[float] = None
    verdict: str = "pending"  # pending | confirmed | rejected
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_hypothesis_block(text: str) -> Dict[str, str]:
    """Extract Hypothesis / ExpectedEffect lines from Coder output or feedback."""
    hyp, exp = "", ""
    for line in (text or "").splitlines():
        s = line.strip()
        low = s.lower()
        if low.startswith("hypothesis:"):
            hyp = s.split(":", 1)[1].strip()
        elif low.startswith("expectedeffect:") or low.startswith("expected_effect:"):
            exp = s.split(":", 1)[1].strip()
    return {"hypothesis": hyp, "expected_effect": exp}


def append_hypothesis(run_dir: PathLike, record: HypothesisRecord) -> Path:
    path = Path(run_dir) / MEMORY_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_hypotheses(run_dir: PathLike) -> List[HypothesisRecord]:
    path = Path(run_dir) / MEMORY_NAME
    if not path.is_file():
        return []
    out: List[HypothesisRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(HypothesisRecord(**{k: d[k] for k in HypothesisRecord.__dataclass_fields__ if k in d}))
    return out


def resolve_pending(
    run_dir: PathLike,
    *,
    metric_after: float,
    min_delta: float = 0.02,
) -> Optional[HypothesisRecord]:
    """Mark the latest pending hypothesis confirmed/rejected from metric delta."""
    rows = load_hypotheses(run_dir)
    pending = [r for r in rows if r.verdict == "pending"]
    if not pending:
        return None
    last = pending[-1]
    before = float(last.metric_before) if last.metric_before is not None else 0.0
    delta = float(metric_after) - before
    last.metric_after = float(metric_after)
    if delta >= float(min_delta):
        last.verdict = "confirmed"
        last.notes = f"delta={delta:+.4f} >= {min_delta}"
    else:
        last.verdict = "rejected"
        last.notes = f"delta={delta:+.4f} < {min_delta}"
    append_hypothesis(run_dir, last)
    return last


def rejected_memory_block(run_dir: PathLike, *, max_items: int = 5) -> str:
    rejected = [r for r in load_hypotheses(run_dir) if r.verdict == "rejected"]
    if not rejected:
        return ""
    lines = ["# Rejected hypotheses (do not repeat):"]
    for r in rejected[-max_items:]:
        lines.append(f"- iter {r.iteration}: {r.hypothesis} ({r.notes})")
    return "\n".join(lines)


def hypothesis_prompt_instruction() -> str:
    return (
        "Before writing code, state exactly two lines:\n"
        "Hypothesis: <one sentence>\n"
        "ExpectedEffect: <metric direction, e.g. +success or +composite>\n"
        "Then provide the reward in a ```python``` block."
    )
