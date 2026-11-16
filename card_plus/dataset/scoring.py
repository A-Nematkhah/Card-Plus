"""Calibrated CrowdNav reward scoring (not just offline rank accuracy).

Pairwise ranking on fixed policies often sits at 0.96–1.0 even for rewards that
fail online. This module builds a composite that also rewards:
  - **separation strength**: how large mean(success) is vs fail (Cohen-like soft gap)
  - **anti-hack margins**: progress/success must beat flee and orbit probes
  - **intra-scenario ranking**: within each scenario, reward order should match
    success≻timeout≻collision (and shorter success preferred) — EvoNav-style

``offline_rank`` is the old weighted pair/AUC score (kept for diagnostics).
``composite`` mixes rank + separation + anti-hack + intra so scores spread out.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from card_plus.dataset.labels import label_episode
from card_plus.dataset.rescore import RescoreReport
from card_plus.dataset.schema import Episode

_OUTCOME_RANK = {
    "success": 3.0,
    "timeout": 2.0,
    "truncated": 1.0,
    "collided": 0.0,
    "unknown": -1.0,
}


def _sigmoid(z: float) -> float:
    z = float(np.clip(z, -20.0, 20.0))
    return float(1.0 / (1.0 + math.exp(-z)))


def soft_gap_score(mean_better: float, mean_worse: float, scale: float) -> float:
    """Map mean_better - mean_worse into (0, 1); 0.5 when equal."""
    s = max(float(scale), 1e-3)
    return _sigmoid((float(mean_better) - float(mean_worse)) / s)


def outcome_group_means(
    episodes: Sequence[Episode],
    returns: Sequence[float],
) -> Dict[str, float]:
    buckets: Dict[str, List[float]] = {}
    for ep, r in zip(episodes, returns):
        outcome = (ep.labels or label_episode(ep)).outcome
        buckets.setdefault(str(outcome), []).append(float(r))
    return {k: float(np.mean(v)) for k, v in buckets.items() if v}


def separation_scores(
    episodes: Sequence[Episode],
    returns: Sequence[float],
    *,
    scale_shrink: float = 0.35,
) -> Dict[str, float]:
    """Soft separation of outcome mean returns.

    ``scale_shrink < 1`` makes the sigmoid steeper so modest mean-gaps no longer
    all collapse near ~0.7 (better discrimination across candidates).
    """
    rets = np.asarray(list(returns), dtype=np.float64)
    scale = float(np.std(rets)) if rets.size > 1 else 1.0
    if scale < 1e-6:
        scale = max(float(np.mean(np.abs(rets))), 1.0)
    scale = max(scale * float(scale_shrink), 1e-3)
    means = outcome_group_means(episodes, returns)
    out: Dict[str, float] = {
        "return_std": float(np.std(rets)) if rets.size > 1 else 0.0,
        "sep_scale": scale,
        "mean_success": means.get("success", float("nan")),
        "mean_collided": means.get("collided", float("nan")),
        "mean_timeout": means.get("timeout", float("nan")),
    }
    if "success" in means and "collided" in means:
        out["sep_success_collided"] = soft_gap_score(
            means["success"], means["collided"], scale
        )
        out["gap_success_collided"] = float(means["success"] - means["collided"])
    else:
        out["sep_success_collided"] = float("nan")
        out["gap_success_collided"] = float("nan")
    if "success" in means and "timeout" in means:
        out["sep_success_timeout"] = soft_gap_score(
            means["success"], means["timeout"], scale
        )
        out["gap_success_timeout"] = float(means["success"] - means["timeout"])
    else:
        out["sep_success_timeout"] = float("nan")
        out["gap_success_timeout"] = float("nan")
    if "timeout" in means and "collided" in means:
        out["sep_timeout_collided"] = soft_gap_score(
            means["timeout"], means["collided"], scale
        )
        out["gap_timeout_collided"] = float(means["timeout"] - means["collided"])
    else:
        out["sep_timeout_collided"] = float("nan")
        out["gap_timeout_collided"] = float("nan")

    vals = [
        out[k]
        for k in ("sep_success_collided", "sep_success_timeout", "sep_timeout_collided")
        if out[k] == out[k]
    ]
    out["separation"] = float(np.mean(vals)) if vals else 0.0
    return out


def rule_quality_score(ep: Episode) -> float:
    """Scalar preference for EvoNav-style within-scenario ranking.

    success ≻ timeout ≻ collision; among successes prefer shorter episodes;
    among failures prefer more goal progress.
    """
    lab = ep.labels or label_episode(ep)
    outcome = str(lab.outcome)
    base = float(_OUTCOME_RANK.get(outcome, -1.0))
    progress = float(getattr(lab, "progress", 0.0) or 0.0)
    t = max(int(ep.T), 1)
    if outcome == "success":
        return base * 1000.0 + progress * 10.0 - float(t)
    if outcome == "timeout":
        return base * 1000.0 + progress * 100.0 - 0.1 * float(t)
    return base * 1000.0 + progress * 50.0 - 0.05 * float(t)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    if float(np.std(rx)) < 1e-12 or float(np.std(ry)) < 1e-12:
        return 0.0
    corr = float(np.corrcoef(rx, ry)[0, 1])
    if corr != corr:
        return 0.0
    return corr


def intra_scenario_score(
    episodes: Sequence[Episode],
    returns: Sequence[float],
) -> Dict[str, float]:
    """Mean Spearman(rule ranks, reward ranks) within each scenario → [0, 1]."""
    by_id: Dict[str, List[Tuple[Episode, float]]] = defaultdict(list)
    for ep, ret in zip(episodes, returns):
        sid = str(getattr(ep.scenario, "scenario_id", "") or "")
        if not sid:
            continue
        by_id[sid].append((ep, float(ret)))

    corrs: List[float] = []
    n_used = 0
    for group in by_id.values():
        if len(group) < 2:
            continue
        rule = np.asarray([rule_quality_score(ep) for ep, _ in group], dtype=np.float64)
        rew = np.asarray([ret for _, ret in group], dtype=np.float64)
        s = _spearman(rule, rew)
        if s != s:
            continue
        corrs.append(s)
        n_used += 1

    if not corrs:
        return {
            "intra_scenario": 0.0,
            "intra_spearman_raw": float("nan"),
            "intra_n_scenarios": 0.0,
        }
    raw = float(np.mean(corrs))
    # map [-1, 1] → [0, 1]
    mapped = 0.5 * (raw + 1.0)
    return {
        "intra_scenario": float(mapped),
        "intra_spearman_raw": raw,
        "intra_n_scenarios": float(n_used),
    }


def anti_hack_score(probe_metrics: Mapping[str, float]) -> float:
    """Continuous anti-hack quality in (0, 1) from flee/orbit probe means."""
    progress = float(probe_metrics.get("progress_mean", 0.0))
    flee = float(probe_metrics.get("flee_mean", 0.0))
    orbit = float(probe_metrics.get("orbit_mean", 0.0))
    success = float(probe_metrics.get("success_mean", 0.0))
    scale = max(abs(progress), abs(flee), abs(orbit), abs(success), 0.5)
    # progress must beat flee/orbit; success must beat flee
    s_pf = soft_gap_score(progress, flee, scale)
    s_po = soft_gap_score(progress, orbit, scale)
    s_sf = soft_gap_score(success, flee, scale)
    # sum gap bonus: progress_sum > flee_sum
    psum = float(probe_metrics.get("progress_sum", 0.0))
    fsum = float(probe_metrics.get("flee_sum", 0.0))
    s_sum = soft_gap_score(psum, fsum, max(abs(psum), abs(fsum), 1.0))
    return float(0.35 * s_pf + 0.25 * s_po + 0.25 * s_sf + 0.15 * s_sum)


def offline_rank_score(
    report: RescoreReport,
    *,
    w_success_collided: float = 0.35,
    w_success_timeout: float = 0.25,
    w_timeout_collided: float = 0.25,
    w_progress_auc: float = 0.15,
) -> float:
    """Legacy weighted pair-accuracy + progress_auc (often ~0.95–1.0)."""
    pairs = {pa.pair: float(pa.accuracy) for pa in report.pair_accuracies if pa.n_pairs > 0}
    a = pairs.get("success/collided", float("nan"))
    b = pairs.get("success/timeout", float("nan"))
    c = pairs.get("timeout/collided", float("nan"))
    d = float(report.progress_auc)
    vals: List[float] = []
    weights: List[float] = []
    for v, w in (
        (a, w_success_collided),
        (b, w_success_timeout),
        (c, w_timeout_collided),
        (d, w_progress_auc),
    ):
        if v == v:
            vals.append(float(v))
            weights.append(float(w))
    if not vals:
        return 0.0
    return float(sum(v * w for v, w in zip(vals, weights)) / sum(weights))


def calibrated_composite(
    *,
    offline_rank: float,
    separation: float,
    anti_hack: float,
    intra_scenario: float = 0.0,
    w_rank: float = 0.10,
    w_sep: float = 0.30,
    w_hack: float = 0.30,
    w_intra: float = 0.30,
) -> float:
    """Mix signals so pairwise-only rewards cannot sit at ~1.0 forever.

    Rank weight is kept low because it saturates near 1.0; separation, anti-hack,
    and intra-scenario ranking provide most of the discrimination.
    """
    parts = [
        (float(offline_rank), float(w_rank)),
        (float(separation), float(w_sep)),
        (float(anti_hack), float(w_hack)),
        (float(intra_scenario), float(w_intra)),
    ]
    vals = [(v, w) for v, w in parts if v == v]
    if not vals:
        return 0.0
    return float(sum(v * w for v, w in vals) / sum(w for _, w in vals))


def build_calibrated_metrics(
    report: RescoreReport,
    episodes: Sequence[Episode],
    returns: Sequence[float],
    probe_metrics: Mapping[str, float],
    *,
    w_rank: float = 0.10,
    w_sep: float = 0.30,
    w_hack: float = 0.30,
    w_intra: float = 0.30,
) -> Tuple[float, Dict[str, float]]:
    """Return (composite, metrics dict including breakdown)."""
    rank = offline_rank_score(report)
    sep_map = separation_scores(episodes, returns)
    intra_map = intra_scenario_score(episodes, returns)
    hack = anti_hack_score(probe_metrics)
    sep = float(sep_map.get("separation", 0.0))
    intra = float(intra_map.get("intra_scenario", 0.0))
    composite = calibrated_composite(
        offline_rank=rank,
        separation=sep,
        anti_hack=hack,
        intra_scenario=intra,
        w_rank=w_rank,
        w_sep=w_sep,
        w_hack=w_hack,
        w_intra=w_intra,
    )
    metrics: Dict[str, float] = {
        "offline_rank": rank,
        "separation": sep,
        "anti_hack": hack,
        "intra_scenario": intra,
        "composite": composite,
        "progress_auc": float(report.progress_auc),
        "tpe_v1_rate": float("nan")
        if report.tpe_v1_rate is None
        else float(report.tpe_v1_rate),
    }
    for pa in report.pair_accuracies:
        metrics[pa.pair] = float(pa.accuracy) if pa.n_pairs > 0 else float("nan")
    metrics.update({k: float(v) for k, v in sep_map.items()})
    metrics.update({k: float(v) for k, v in intra_map.items()})
    metrics.update({k: float(v) for k, v in probe_metrics.items()})
    return composite, metrics
