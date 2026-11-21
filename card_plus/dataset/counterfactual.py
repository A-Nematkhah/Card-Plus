"""Counterfactual ablation of reward components on stored episodes."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from card_plus.dataset.attribution import episode_component_sums
from card_plus.dataset.rescore import (
    RewardFn,
    _pair_accuracy,
    episode_returns_for_reward,
    load_reward_fn_from_code,
)
from card_plus.dataset.schema import Episode

PathLike = Union[str, Path]

_DEFAULT_PAIRS = (
    ("success", "collided"),
    ("success", "timeout"),
    ("timeout", "collided"),
)


@dataclass
class CounterfactualRow:
    component: str
    mode: str  # drop | scale2
    pair_deltas: Dict[str, float] = field(default_factory=dict)
    harmful: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "mode": self.mode,
            "pair_deltas": dict(self.pair_deltas),
            "harmful": self.harmful,
        }


@dataclass
class CounterfactualReport:
    baseline_pairs: Dict[str, float]
    rows: List[CounterfactualRow]
    flagged: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_pairs": dict(self.baseline_pairs),
            "rows": [r.to_dict() for r in self.rows],
            "flagged": list(self.flagged),
        }


def _wrap_component(
    reward_fn: RewardFn,
    component: str,
    *,
    mode: str,
) -> RewardFn:
    """Return a reward that drops or ×2 one component in the returned dict."""

    def _call(action: np.ndarray, obs: np.ndarray) -> Any:
        result = reward_fn(action, obs)
        if not (isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], dict)):
            return result
        total = float(result[0])
        comps = dict(result[1])
        if component not in comps:
            return result
        old = float(comps[component])
        if mode == "drop":
            comps[component] = 0.0
            total = total - old
        elif mode == "scale2":
            comps[component] = old * 2.0
            total = total + old
        else:
            raise ValueError(f"unknown mode {mode!r}")
        return total, comps

    set_pre = getattr(reward_fn, "set_pre_obs", None)
    reset = getattr(reward_fn, "reset_pre_obs", None)
    if set_pre is not None:
        _call.set_pre_obs = set_pre  # type: ignore[attr-defined]
    if reset is not None:
        _call.reset_pre_obs = reset  # type: ignore[attr-defined]
    return _call


def _pair_map(
    episodes: Sequence[Episode],
    returns: Sequence[float],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for a, b in _DEFAULT_PAIRS:
        pa = _pair_accuracy(episodes, returns, a, b)
        key = f"{a}/{b}"
        out[key] = float(pa.accuracy) if pa.n_pairs > 0 else float("nan")
    return out


def discover_components(
    episodes: Sequence[Episode],
    reward_fn: RewardFn,
    *,
    max_episodes: int = 8,
) -> List[str]:
    keys = set()
    for ep in list(episodes)[: max(1, int(max_episodes))]:
        _t, comps = episode_component_sums(ep, reward_fn)
        keys.update(comps.keys())
    return sorted(keys)


def run_counterfactual(
    episodes: Sequence[Episode],
    reward_fn: RewardFn,
    *,
    components: Optional[Sequence[str]] = None,
    harm_eps: float = 0.02,
) -> CounterfactualReport:
    """Drop / ×2 each component; flag ones that hurt pair ordering."""
    base_returns = episode_returns_for_reward(episodes, reward_fn)
    baseline = _pair_map(episodes, base_returns)
    comps = list(components) if components is not None else discover_components(episodes, reward_fn)
    rows: List[CounterfactualRow] = []
    flagged: List[str] = []
    for name in comps:
        for mode in ("drop", "scale2"):
            wrapped = _wrap_component(reward_fn, name, mode=mode)
            rets = episode_returns_for_reward(episodes, wrapped)
            pairs = _pair_map(episodes, rets)
            deltas: Dict[str, float] = {}
            harmful = False
            for key, base_v in baseline.items():
                new_v = pairs.get(key, float("nan"))
                if base_v != base_v or new_v != new_v:
                    deltas[key] = float("nan")
                    continue
                d = float(new_v - base_v)
                deltas[key] = d
                if d < -float(harm_eps):
                    harmful = True
            row = CounterfactualRow(
                component=name, mode=mode, pair_deltas=deltas, harmful=harmful
            )
            rows.append(row)
            if harmful and name not in flagged:
                flagged.append(name)
    return CounterfactualReport(baseline_pairs=baseline, rows=rows, flagged=flagged)


def run_counterfactual_from_code(
    episodes: Sequence[Episode],
    reward_code: str,
    **kwargs: Any,
) -> CounterfactualReport:
    return run_counterfactual(episodes, load_reward_fn_from_code(reward_code), **kwargs)


def write_counterfactual_report(
    report: CounterfactualReport,
    path: PathLike,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return p
