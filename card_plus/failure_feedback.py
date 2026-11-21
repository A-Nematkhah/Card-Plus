"""Phase-5: structured failure-mode feedback from labeled episodes / trajs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from card_plus.dataset.labels import label_episode
from card_plus.dataset.schema import Episode


def cluster_failure_tags(episodes: Sequence[Episode]) -> Dict[str, int]:
    counts: Counter = Counter()
    for ep in episodes:
        labels = ep.labels or label_episode(ep)
        if labels.outcome == "success":
            continue
        tags = list(labels.failure_tags or [])
        if not tags:
            tags = [str(labels.outcome)]
        for t in tags:
            counts[str(t)] += 1
    return dict(counts)


def first_divergence_step(
    success_obs: np.ndarray,
    fail_obs: np.ndarray,
    *,
    goal_slice: slice = slice(4, 6),
    pos_slice: slice = slice(0, 2),
) -> Optional[int]:
    """First timestep where fail traj is clearly farther from goal than success."""
    t = int(min(len(success_obs), len(fail_obs)))
    if t == 0:
        return None
    for i in range(t):
        ds = float(np.linalg.norm(success_obs[i, pos_slice] - success_obs[i, goal_slice]))
        df = float(np.linalg.norm(fail_obs[i, pos_slice] - fail_obs[i, goal_slice]))
        if df > ds + 0.5:
            return int(i)
    return None


def format_failure_feedback(
    episodes: Sequence[Episode],
    *,
    max_lines: int = 12,
) -> str:
    counts = cluster_failure_tags(episodes)
    if not counts:
        return ""
    total_fail = sum(counts.values())
    lines = ["# Failure mode clusters (offline labels):"]
    for tag, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        share = 100.0 * float(n) / float(max(total_fail, 1))
        lines.append(f"- {tag}: n={n} ({share:.0f}%)")
        if len(lines) >= max_lines:
            break
    # Optional divergence hint using one success + one fail.
    succ = next((e for e in episodes if (e.labels or label_episode(e)).outcome == "success"), None)
    fail = next(
        (e for e in episodes if (e.labels or label_episode(e)).outcome != "success"),
        None,
    )
    if succ is not None and fail is not None and succ.T > 0 and fail.T > 0:
        step = first_divergence_step(succ.obs_array(), fail.obs_array())
        if step is not None:
            lines.append(f"- first_divergence_step≈{step} (success vs fail goal-distance)")
    return "\n".join(lines[:max_lines])
