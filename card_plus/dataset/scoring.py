"""Calibrated CrowdNav reward scoring (not just offline rank accuracy).

Pairwise ranking on fixed policies often sits at 0.96–1.0 even for rewards that
fail online. This module builds a composite that also rewards:
  - **separation strength**: how large mean(success) is vs fail (Cohen-like soft gap)
  - **anti-hack margins**: progress/success must beat flee and orbit probes

``offline_rank`` is the old weighted pair/AUC score (kept for diagnostics).
``composite`` mixes rank + separation + anti-hack so scores spread out.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from card_plus.dataset.labels import label_episode
from card_plus.dataset.rescore import RescoreReport
from card_plus.dataset.schema import Episode


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
) -> Dict[str, float]:
    """Soft separation of outcome mean returns (scale = pooled std)."""
    rets = np.asarray(list(returns), dtype=np.float64)
    scale = float(np.std(rets)) if rets.size > 1 else 1.0
    if scale < 1e-6:
        scale = max(float(np.mean(np.abs(rets))), 1.0)
    means = outcome_group_means(episodes, returns)
    out: Dict[str, float] = {
        "return_std": scale,
        "mean_success": means.get("success", float("nan")),
        "mean_collided": means.get("collided", float("nan")),
        "mean_timeout": means.get("timeout", float("nan")),
    }
    if "success" in means and "collided" in means:
        out["sep_success_collided"] = soft_gap_score(
            means["success"], means["collided"], scale
        )
    else:
        out["sep_success_collided"] = float("nan")
    if "success" in means and "timeout" in means:
        out["sep_success_timeout"] = soft_gap_score(
            means["success"], means["timeout"], scale
        )
    else:
        out["sep_success_timeout"] = float("nan")
    if "timeout" in means and "collided" in means:
        out["sep_timeout_collided"] = soft_gap_score(
            means["timeout"], means["collided"], scale
        )
    else:
        out["sep_timeout_collided"] = float("nan")

    vals = [
        out[k]
        for k in ("sep_success_collided", "sep_success_timeout", "sep_timeout_collided")
        if out[k] == out[k]
    ]
    out["separation"] = float(np.mean(vals)) if vals else 0.0
    return out


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
    w_rank: float = 0.25,
    w_sep: float = 0.35,
    w_hack: float = 0.40,
) -> float:
    """Mix signals so pairwise-only rewards cannot sit at ~1.0 forever."""
    parts = [
        (float(offline_rank), float(w_rank)),
        (float(separation), float(w_sep)),
        (float(anti_hack), float(w_hack)),
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
    w_rank: float = 0.25,
    w_sep: float = 0.35,
    w_hack: float = 0.40,
) -> Tuple[float, Dict[str, float]]:
    """Return (composite, metrics dict including breakdown)."""
    rank = offline_rank_score(report)
    sep_map = separation_scores(episodes, returns)
    hack = anti_hack_score(probe_metrics)
    sep = float(sep_map.get("separation", 0.0))
    composite = calibrated_composite(
        offline_rank=rank,
        separation=sep,
        anti_hack=hack,
        w_rank=w_rank,
        w_sep=w_sep,
        w_hack=w_hack,
    )
    metrics: Dict[str, float] = {
        "offline_rank": rank,
        "separation": sep,
        "anti_hack": hack,
        "composite": composite,
        "progress_auc": float(report.progress_auc),
        "tpe_v1_rate": float("nan")
        if report.tpe_v1_rate is None
        else float(report.tpe_v1_rate),
    }
    for pa in report.pair_accuracies:
        metrics[pa.pair] = float(pa.accuracy) if pa.n_pairs > 0 else float("nan")
    metrics.update({k: float(v) for k, v in sep_map.items()})
    metrics.update({k: float(v) for k, v in probe_metrics.items()})
    return composite, metrics
