"""Offline sub-reward attribution for failed outcome pairs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from card_plus.dataset.labels import label_episode
from card_plus.dataset.rescore import RewardFn, ensure_episode_gate_arrays
from card_plus.dataset.schema import Episode


def _component_dict(result: Any) -> Dict[str, float]:
    if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], dict):
        out: Dict[str, float] = {}
        for k, v in result[1].items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return out
    return {}


def episode_component_sums(
    episode: Episode,
    reward_fn: RewardFn,
) -> Tuple[float, Dict[str, float]]:
    """Return (total_sum, {component: sum}) for one episode."""
    ensure_episode_gate_arrays(episode)
    actions = episode._gate_actions  # type: ignore[attr-defined]
    obs = episode._gate_obs  # type: ignore[attr-defined]
    initial = episode._gate_initial  # type: ignore[attr-defined]
    reset = getattr(reward_fn, "reset_pre_obs", None)
    set_pre = getattr(reward_fn, "set_pre_obs", None)
    if reset is not None:
        reset()
    totals = 0.0
    comps: Dict[str, float] = defaultdict(float)
    prev = None if initial is None else initial
    n = int(actions.shape[0])
    for t in range(n):
        obs_t = obs[t]
        if set_pre is not None:
            set_pre(prev if prev is not None else obs_t)
        result = reward_fn(actions[t], obs_t)
        if isinstance(result, tuple):
            totals += float(result[0])
        else:
            totals += float(result)
        for k, v in _component_dict(result).items():
            comps[k] += float(v)
        prev = obs_t
    return float(totals), dict(comps)


def mean_components_by_outcome(
    episodes: Sequence[Episode],
    reward_fn: RewardFn,
) -> Dict[str, Dict[str, float]]:
    """Mean episode-sum of each sub-reward keyed by outcome label."""
    buckets: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for ep in episodes:
        outcome = str((ep.labels or label_episode(ep)).outcome)
        _total, comps = episode_component_sums(ep, reward_fn)
        buckets[outcome].append(comps)
    out: Dict[str, Dict[str, float]] = {}
    for outcome, rows in buckets.items():
        keys = sorted({k for row in rows for k in row})
        means: Dict[str, float] = {}
        for k in keys:
            vals = [float(row.get(k, 0.0)) for row in rows]
            means[k] = float(np.mean(vals)) if vals else 0.0
        means["_n"] = float(len(rows))
        out[outcome] = means
    return out


def pair_attribution(
    episodes: Sequence[Episode],
    reward_fn: RewardFn,
    pair: str,
) -> Dict[str, Any]:
    """Compare mean component sums for the two outcomes in ``a/b`` pair name."""
    if "/" not in pair:
        raise ValueError(f"pair must look like 'timeout/collided', got {pair!r}")
    better, worse = pair.split("/", 1)
    by_out = mean_components_by_outcome(episodes, reward_fn)
    a = by_out.get(better, {})
    b = by_out.get(worse, {})
    keys = sorted((set(a) | set(b)) - {"_n"})
    rows = []
    for k in keys:
        ma = float(a.get(k, 0.0))
        mb = float(b.get(k, 0.0))
        rows.append(
            {
                "component": k,
                f"mean_{better}": ma,
                f"mean_{worse}": mb,
                "delta_better_minus_worse": ma - mb,
            }
        )
    rows.sort(key=lambda r: abs(float(r["delta_better_minus_worse"])), reverse=True)
    return {
        "pair": pair,
        "n_better": int(a.get("_n", 0.0)),
        "n_worse": int(b.get("_n", 0.0)),
        "components": rows[:12],
    }


def format_attribution_block(
    attributions: Sequence[Mapping[str, Any]],
    *,
    max_pairs: int = 3,
    max_components: int = 5,
) -> str:
    """Compact text/JSON-ish block for LLM feedback."""
    if not attributions:
        return ""
    lines = ["  attribution (mean episode component sums on failed pairs):"]
    for attr in list(attributions)[:max_pairs]:
        pair = attr.get("pair", "?")
        lines.append(
            f"    {pair}: n={attr.get('n_better', 0)}/{attr.get('n_worse', 0)}"
        )
        for row in list(attr.get("components") or [])[:max_components]:
            lines.append(
                f"      {row['component']}: "
                f"delta={float(row['delta_better_minus_worse']):+.3f}"
            )
    return "\n".join(lines)


def failed_pair_attributions(
    episodes: Sequence[Episode],
    reward_fn: RewardFn,
    *,
    pair_metrics: Mapping[str, float],
    floors: Mapping[str, float],
) -> List[Dict[str, Any]]:
    """Attribution only for pairs that miss their floor (or are NaN)."""
    out: List[Dict[str, Any]] = []
    for pair, floor in floors.items():
        val = float(pair_metrics.get(pair, float("nan")))
        if val != val or val < float(floor):
            out.append(pair_attribution(episodes, reward_fn, pair))
    return out
