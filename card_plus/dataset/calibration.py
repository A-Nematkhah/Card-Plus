"""Phase-2 calibration: composite↔online correlation and difficulty reports."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from card_plus.dataset.anti_hack_probes import run_anti_hack_probes
from card_plus.dataset.rescore import (
    RescoreReport,
    episode_returns_for_reward,
    load_reward_fn_from_code,
    rescore_episodes_detailed,
)
from card_plus.dataset.scoring import build_calibrated_metrics, calibrated_composite
from card_plus.dataset.schema import Episode
from card_plus.dataset.split import difficulty_bucket
from card_plus.tpe_v2 import spearman_rho


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall τ; constant / short inputs → 0.0."""
    a = np.asarray(list(x), dtype=np.float64).ravel()
    b = np.asarray(list(y), dtype=np.float64).ravel()
    if a.size < 2 or a.size != b.size:
        return 0.0
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    try:
        from scipy.stats import kendalltau

        tau, _ = kendalltau(a, b)
        if tau is None or (isinstance(tau, float) and tau != tau):
            return 0.0
        return float(tau)
    except Exception:
        # O(n^2) sign concordance fallback
        n = int(a.size)
        conc = 0
        disc = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                da = a[i] - a[j]
                db = b[i] - b[j]
                if da == 0.0 or db == 0.0:
                    continue
                if da * db > 0:
                    conc += 1
                else:
                    disc += 1
        denom = conc + disc
        return float(conc - disc) / float(denom) if denom else 0.0


@dataclass
class ZooEntryScore:
    reward_id: str
    composite: float
    metrics: Dict[str, float]
    online_success: Optional[float] = None
    online_proxy: Optional[float] = None
    label_name: str = "online_success"
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward_id": self.reward_id,
            "composite": self.composite,
            "metrics": dict(self.metrics),
            "online_success": self.online_success,
            "online_proxy": self.online_proxy,
            "label_name": self.label_name,
            "source": self.source,
        }

    def calibration_target(self) -> Optional[float]:
        """Prefer continuous proxy when success labels are missing/degenerate."""
        if self.online_proxy is not None:
            return float(self.online_proxy)
        if self.online_success is not None:
            return float(self.online_success)
        return None


@dataclass
class CalibrationReport:
    n: int
    spearman: float
    kendall: float
    go: bool
    threshold: float
    entries: List[ZooEntryScore] = field(default_factory=list)
    component_ablation: Dict[str, float] = field(default_factory=dict)
    difficulty_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "spearman": self.spearman,
            "kendall": self.kendall,
            "go": self.go,
            "threshold": self.threshold,
            "entries": [e.to_dict() for e in self.entries],
            "component_ablation": dict(self.component_ablation),
            "difficulty_metrics": {
                k: dict(v) for k, v in self.difficulty_metrics.items()
            },
            "notes": list(self.notes),
        }


def score_reward_on_episodes(
    reward_code: str,
    episodes: Sequence[Episode],
    *,
    use_suicide_probe: bool = False,
    learned_judge=None,
) -> Tuple[float, Dict[str, float], RescoreReport]:
    fn = load_reward_fn_from_code(reward_code)
    report, labeled, returns = rescore_episodes_detailed(episodes, fn, reward_code=reward_code)
    probe = run_anti_hack_probes(fn, use_suicide_probe=use_suicide_probe)
    composite, metrics = build_calibrated_metrics(
        report, labeled, returns, probe.metrics, learned_judge=learned_judge
    )
    return float(composite), metrics, report


def composite_component_ablation(
    entries: Sequence[ZooEntryScore],
) -> Dict[str, float]:
    """Spearman(composite_without_component, online) for each dropped weight.

    Requires entries that already store per-component metrics + online_success.
    Recomputes composite with one weight zeroed.
    """
    labeled = [e for e in entries if e.online_success is not None]
    if len(labeled) < 3:
        return {}
    online = [float(e.online_success) for e in labeled]  # type: ignore[arg-type]
    out: Dict[str, float] = {}
    weight_sets = {
        "full": dict(w_rank=0.05, w_sep=0.15, w_hack=0.65, w_intra=0.15),
        "drop_rank": dict(w_rank=0.0, w_sep=0.15, w_hack=0.65, w_intra=0.15),
        "drop_sep": dict(w_rank=0.05, w_sep=0.0, w_hack=0.65, w_intra=0.15),
        "drop_anti_hack": dict(w_rank=0.05, w_sep=0.15, w_hack=0.0, w_intra=0.15),
        "drop_intra": dict(w_rank=0.05, w_sep=0.15, w_hack=0.65, w_intra=0.0),
    }
    for name, weights in weight_sets.items():
        comps = []
        for e in labeled:
            m = e.metrics
            comps.append(
                calibrated_composite(
                    offline_rank=float(m.get("offline_rank", float("nan"))),
                    separation=float(m.get("separation", float("nan"))),
                    anti_hack=float(m.get("anti_hack", float("nan"))),
                    intra_scenario=float(m.get("intra_scenario", float("nan"))),
                    **weights,
                )
            )
        out[name] = float(spearman_rho(np.asarray(comps), np.asarray(online)))
    return out


def metrics_by_difficulty(
    episodes: Sequence[Episode],
    returns: Sequence[float],
    report_factory,
) -> Dict[str, Dict[str, float]]:
    """Group episodes by difficulty bucket and compute pair/AUC metrics.

    ``report_factory(eps, rets) -> dict`` should return metric floats.
    """
    buckets: Dict[str, List[int]] = defaultdict(list)
    for i, ep in enumerate(episodes):
        buckets[difficulty_bucket(ep)].append(i)
    out: Dict[str, Dict[str, float]] = {}
    for name, idxs in sorted(buckets.items()):
        if len(idxs) < 2:
            out[name] = {"n": float(len(idxs))}
            continue
        sub_eps = [episodes[i] for i in idxs]
        sub_rets = [returns[i] for i in idxs]
        metrics = report_factory(sub_eps, sub_rets)
        metrics["n"] = float(len(idxs))
        out[name] = metrics
    return out


def default_pair_metrics(
    episodes: Sequence[Episode],
    returns: Sequence[float],
) -> Dict[str, float]:
    from card_plus.dataset.rescore import (
        outcome_return_auc,
        _pair_accuracy,
    )

    metrics: Dict[str, float] = {
        "progress_auc": float(outcome_return_auc(episodes, returns)),
        "mean_return": float(np.mean(returns)) if returns else float("nan"),
    }
    for a, b in (("success", "collided"), ("success", "timeout"), ("timeout", "collided")):
        pa = _pair_accuracy(episodes, returns, a, b)
        key = f"{a}/{b}"
        metrics[key] = float(pa.accuracy) if pa.n_pairs > 0 else float("nan")
    return metrics


def calibrate_reward_zoo(
    entries: Sequence[ZooEntryScore],
    *,
    spearman_threshold: float = 0.5,
    min_n: int = 20,
    use_proxy: bool = True,
) -> CalibrationReport:
    """Go if Spearman(composite, online target) >= threshold with N >= min_n.

    When ``use_proxy`` is True, uses ``online_proxy`` (progress / version_score /
    ep_rew) falling back to ``online_success``. Constant targets → No-Go note.
    """
    labeled = [e for e in entries if e.calibration_target() is not None]
    notes: List[str] = []
    if len(labeled) < int(min_n):
        notes.append(
            f"N={len(labeled)} < min_n={min_n}: treat as incomplete calibration "
            "(need more online labels)."
        )
    if len(labeled) < 3:
        return CalibrationReport(
            n=len(labeled),
            spearman=0.0,
            kendall=0.0,
            go=False,
            threshold=float(spearman_threshold),
            entries=list(entries),
            notes=notes + ["fewer than 3 labeled entries"],
        )
    comps = np.asarray([e.composite for e in labeled], dtype=np.float64)
    if use_proxy:
        online = np.asarray(
            [float(e.calibration_target()) for e in labeled], dtype=np.float64
        )
    else:
        online = np.asarray(
            [float(e.online_success) for e in labeled if e.online_success is not None],
            dtype=np.float64,
        )
        comps = np.asarray(
            [e.composite for e in labeled if e.online_success is not None],
            dtype=np.float64,
        )
    if online.size >= 3 and float(np.std(online)) < 1e-8:
        notes.append(
            "Online targets are constant (e.g. all-zero success) — Spearman is "
            "meaningless. Import non-zero archive proxies or lengthen training."
        )
        return CalibrationReport(
            n=int(online.size),
            spearman=0.0,
            kendall=0.0,
            go=False,
            threshold=float(spearman_threshold),
            entries=list(entries),
            notes=notes,
        )
    sp = float(spearman_rho(comps, online))
    kd = float(kendall_tau(comps, online))
    go = bool(sp >= float(spearman_threshold) and len(labeled) >= int(min_n))
    if sp < float(spearman_threshold):
        notes.append(
            f"Spearman={sp:.3f} < {spearman_threshold:.2f} → No-Go: redesign scoring "
            "weights from data or learned judge (G2) before Phase 3."
        )
    elif go:
        notes.append(f"Go: Spearman={sp:.3f} with N={len(labeled)}.")
    # Component ablation still keys off online_success; copy proxy into it temporarily.
    abl_entries = []
    for e in labeled:
        abl_entries.append(
            ZooEntryScore(
                reward_id=e.reward_id,
                composite=e.composite,
                metrics=e.metrics,
                online_success=e.calibration_target(),
                online_proxy=e.online_proxy,
                label_name=e.label_name,
                source=e.source,
            )
        )
    ablation = composite_component_ablation(abl_entries)
    return CalibrationReport(
        n=len(labeled),
        spearman=sp,
        kendall=kd,
        go=go,
        threshold=float(spearman_threshold),
        entries=list(entries),
        component_ablation=ablation,
        notes=notes,
    )


def difficulty_report_for_reward(
    reward_code: str,
    episodes: Sequence[Episode],
) -> Dict[str, Dict[str, float]]:
    fn = load_reward_fn_from_code(reward_code)
    returns = episode_returns_for_reward(episodes, fn, reward_code=reward_code)
    return metrics_by_difficulty(episodes, returns, default_pair_metrics)
