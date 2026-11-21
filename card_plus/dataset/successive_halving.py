"""Helpers for successive-halving / ASHA-style selection.

CrowdNav CARD uses *across-step* successive halfing:
  1. Each query: gate evaluates LLM candidates; only the best composite is
     written out (see ``candidate_loop``).
  2. Each CARD train step runs a short budget (e.g. 50k).
  3. After all steps, ``crowdnav_finalists`` trains the top-K archive
     versions for a long budget (e.g. 200k).

Phase-4 adds multi-seed ASHA rungs: each candidate is scored by the median
success across seeds before survivor selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


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
    seed_scores: List[float] = field(default_factory=list)
    score_std: Optional[float] = None
    n_seeds: int = 0


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


def aggregate_seed_scores(scores: Sequence[float]) -> Tuple[float, float, int]:
    """Return (median, std, n). Empty → (0, 0, 0)."""
    arr = np.asarray(list(scores), dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0, 0.0, 0
    med = float(np.median(arr))
    std = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
    return med, std, int(arr.size)


def apply_seed_scores(cand: SHCandidate, scores: Sequence[float]) -> SHCandidate:
    med, std, n = aggregate_seed_scores(scores)
    cand.seed_scores = [float(s) for s in scores]
    cand.success_rate = med
    cand.score_std = std
    cand.n_seeds = n
    return cand


def rank_candidates(cands: Sequence[SHCandidate]) -> List[SHCandidate]:
    """Higher success (median), then lower std, then composite, then earlier slot."""
    return sorted(
        cands,
        key=lambda c: (
            float(c.success_rate),
            -float(c.score_std or 0.0),
            float(c.composite),
            -int(c.slot),
        ),
        reverse=True,
    )


def select_survivors(
    cands: Sequence[SHCandidate], keep: Optional[int] = None
) -> List[SHCandidate]:
    ranked = rank_candidates(cands)
    k = survivors_to_keep(len(ranked)) if keep is None else max(1, int(keep))
    return ranked[: min(k, len(ranked))]


def asha_rung(
    cands: Sequence[SHCandidate],
    *,
    seeds: Sequence[int],
    train_fn: Callable[[SHCandidate, int], float],
    keep: Optional[int] = None,
) -> List[SHCandidate]:
    """Run one ASHA rung: score each cand on ``seeds``, keep top half by median.

    ``train_fn(cand, seed) -> success_rate`` performs the short RL for one seed.
    """
    updated: List[SHCandidate] = []
    seed_list = [int(s) for s in seeds] or [0]
    for cand in cands:
        scores = [float(train_fn(cand, seed)) for seed in seed_list]
        updated.append(apply_seed_scores(cand, scores))
    return select_survivors(updated, keep=keep)


def asha_bracket(
    cands: Sequence[SHCandidate],
    *,
    rung_steps: Sequence[int],
    seeds: Sequence[int],
    train_fn: Callable[[SHCandidate, int, int], float],
) -> SHCandidate:
    """Multi-rung ASHA: at rung i train ``rung_steps[i]`` then halve.

    ``train_fn(cand, seed, steps) -> success``.
    """
    active = list(cands)
    if not active:
        raise ValueError("asha_bracket requires at least one candidate")
    steps_list = parse_rung_steps(rung_steps)
    for steps in steps_list:
        if len(active) <= 1:
            break

        def _fn(c: SHCandidate, seed: int, _steps: int = int(steps)) -> float:
            return float(train_fn(c, seed, _steps))

        active = asha_rung(
            active,
            seeds=seeds,
            train_fn=lambda c, s: _fn(c, s),
        )
    return rank_candidates(active)[0]
