"""Helpers for successive-halving style selection.

CrowdNav CARD uses *across-step* successive halfing:
  1. Each query: gate evaluates LLM candidates; only the best composite is
     written out (see ``candidate_loop``).
  2. Each CARD train step runs a short budget (e.g. 50k).
  3. After all steps, ``crowdnav_finalists`` trains the top-K archive
     versions for a long budget (e.g. 200k).

The pure ranking helpers below are shared/tested; per-candidate short RL
rungs are no longer part of the default CrowdNav loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence


@dataclass
class SHCandidate:
    slot: int
    attempt: int
    specific_code: str
    general_code: str = ""
    composite: float = 0.0
    success_rate: float = 0.0
    model_path: Optional[str] = None
    code_path: Optional[str] = None


def parse_rung_steps(raw: Any, default: Sequence[int] = (50_000,)) -> List[int]:
    """Parse ``\"50000\"`` or ``[50000, 200000]`` into positive ints."""
    if raw is None or raw == "":
        return [int(x) for x in default]
    if isinstance(raw, (list, tuple)):
        steps = [int(x) for x in raw]
    else:
        steps = [int(p.strip()) for p in str(raw).split(",") if p.strip()]
    if not steps or any(s <= 0 for s in steps):
        raise ValueError(f"sh_rung_steps must be positive ints, got {raw!r}")
    return steps


def survivors_to_keep(n_active: int) -> int:
    """Keep top half rounded up so 3→2→1 (not 3→1)."""
    if n_active <= 1:
        return n_active
    return max(1, (n_active + 1) // 2)


def rank_candidates(cands: Sequence[SHCandidate]) -> List[SHCandidate]:
    """Higher success, then composite, then earlier slot."""
    return sorted(
        cands,
        key=lambda c: (float(c.success_rate), float(c.composite), -int(c.slot)),
        reverse=True,
    )


def select_survivors(
    cands: Sequence[SHCandidate], keep: Optional[int] = None
) -> List[SHCandidate]:
    ranked = rank_candidates(cands)
    k = survivors_to_keep(len(ranked)) if keep is None else max(1, int(keep))
    return ranked[: min(k, len(ranked))]
