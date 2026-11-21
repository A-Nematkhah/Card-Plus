"""G2 learned judge: signed linear mix of offline metrics vs online proxy.

Phase-2 evidence: positive-only composite weights stay anti-correlated with
``mean_progress`` (esp. ``intra_scenario``). This module fits signed weights
that maximize Spearman against continuous online targets, with leave-one-out
honesty for tiny N.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from card_plus.tpe_v2 import spearman_rho

# Core gate metrics + a few scale-aware diagnostics (kept small for N≈7–20).
DEFAULT_FEATURES: Tuple[str, ...] = (
    "offline_rank",
    "separation",
    "anti_hack",
    "intra_scenario",
    "progress_mean",
    "sep_scale",
)


@dataclass
class LearnedJudge:
    features: List[str]
    weights: List[float]
    bias: float = 0.0
    score_mean: float = 0.0
    score_std: float = 1.0
    spearman_train: float = 0.0
    spearman_loo: float = 0.0
    n: int = 0
    target_name: str = "online_proxy"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LearnedJudge":
        return cls(
            features=list(d.get("features") or []),
            weights=[float(w) for w in (d.get("weights") or [])],
            bias=float(d.get("bias") or 0.0),
            score_mean=float(d.get("score_mean") or 0.0),
            score_std=float(d.get("score_std") or 1.0),
            spearman_train=float(d.get("spearman_train") or 0.0),
            spearman_loo=float(d.get("spearman_loo") or 0.0),
            n=int(d.get("n") or 0),
            target_name=str(d.get("target_name") or "online_proxy"),
            notes=list(d.get("notes") or []),
        )

    def score_metrics(self, metrics: Mapping[str, float]) -> float:
        total = float(self.bias)
        for name, w in zip(self.features, self.weights):
            v = metrics.get(name)
            if v is None:
                continue
            fv = float(v)
            if fv != fv:  # NaN
                continue
            total += float(w) * fv
        return float(total)

    def gate_score(self, metrics: Mapping[str, float]) -> float:
        """Map raw judge score to ~(0,1) via train z-score sigmoid for gate floors."""
        raw = self.score_metrics(metrics)
        sd = float(self.score_std) if float(self.score_std) > 1e-8 else 1.0
        z = (raw - float(self.score_mean)) / sd
        z = float(max(-20.0, min(20.0, z)))
        return float(1.0 / (1.0 + math.exp(-z)))


def extract_feature_matrix(
    metrics_list: Sequence[Mapping[str, float]],
    features: Sequence[str],
) -> np.ndarray:
    rows: List[List[float]] = []
    for m in metrics_list:
        row = []
        for name in features:
            v = m.get(name)
            fv = float(v) if v is not None else float("nan")
            if fv != fv:
                fv = 0.0
            row.append(fv)
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def _zscore_cols(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (x - mu) / sd, mu, sd


def _spearman_for_weights(
    x_std: np.ndarray, y: np.ndarray, w: np.ndarray
) -> float:
    pred = x_std @ w
    if float(np.std(pred)) < 1e-12:
        return 0.0
    return float(spearman_rho(pred, y))


def fit_signed_weights(
    x: np.ndarray,
    y: np.ndarray,
    *,
    grid: Sequence[float] = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0),
) -> Tuple[np.ndarray, float, float]:
    """Coordinate-wise grid search for signed weights maximizing Spearman.

    Returns ``(w_raw, bias, spearman)`` so ``score = x @ w_raw + bias``.
    """
    x_std, mu, sd = _zscore_cols(x)
    d = x_std.shape[1]
    w = np.zeros(d, dtype=np.float64)
    best = _spearman_for_weights(x_std, y, w)
    for _ in range(4):
        improved = False
        for j in range(d):
            local_best = best
            local_wj = float(w[j])
            for cand in grid:
                w[j] = float(cand)
                sp = _spearman_for_weights(x_std, y, w)
                if sp > local_best + 1e-9:
                    local_best = sp
                    local_wj = float(cand)
                    improved = True
            w[j] = local_wj
            best = local_best
        if not improved:
            break
    w_raw = w / sd
    bias = float(-np.dot(mu, w_raw))
    return w_raw, bias, float(best)


def leave_one_out_spearman(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    n = int(y.shape[0])
    if n < 4:
        return float("nan")
    preds = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        w_raw, bias, _ = fit_signed_weights(x[mask], y[mask])
        preds[i] = float(np.dot(x[i], w_raw) + bias)
    if np.isnan(preds).any() or float(np.std(preds)) < 1e-12:
        return float("nan")
    return float(spearman_rho(preds, y))


def fit_learned_judge(
    metrics_list: Sequence[Mapping[str, float]],
    targets: Sequence[float],
    *,
    features: Sequence[str] = DEFAULT_FEATURES,
    target_name: str = "online_proxy",
) -> LearnedJudge:
    y = np.asarray([float(t) for t in targets], dtype=np.float64)
    notes: List[str] = []
    if y.size < 3:
        return LearnedJudge(
            features=list(features),
            weights=[0.0] * len(features),
            n=int(y.size),
            target_name=target_name,
            notes=["need >=3 labeled rows"],
        )
    if float(np.std(y)) < 1e-8:
        return LearnedJudge(
            features=list(features),
            weights=[0.0] * len(features),
            n=int(y.size),
            target_name=target_name,
            notes=["targets constant — cannot fit"],
        )
    # Drop features that are constant on this sample.
    x_full = extract_feature_matrix(metrics_list, features)
    keep_idx = [j for j in range(x_full.shape[1]) if float(np.std(x_full[:, j])) >= 1e-8]
    used = [features[j] for j in keep_idx]
    if not used:
        return LearnedJudge(
            features=list(features),
            weights=[0.0] * len(features),
            n=int(y.size),
            target_name=target_name,
            notes=["all features constant"],
        )
    x = x_full[:, keep_idx]
    w_raw, bias, sp_train = fit_signed_weights(x, y)
    loo = leave_one_out_spearman(x, y)
    raw_preds = x @ w_raw + bias
    notes.append(
        f"train Spearman={sp_train:.3f}; LOO Spearman={loo:.3f} (N={y.size}). "
        "Treat LOO as honesty check; do not claim Phase-2 Go from train alone."
    )
    return LearnedJudge(
        features=list(used),
        weights=[float(w) for w in w_raw],
        bias=float(bias),
        score_mean=float(np.mean(raw_preds)),
        score_std=float(max(float(np.std(raw_preds)), 1e-6)),
        spearman_train=float(sp_train),
        spearman_loo=float(loo) if loo == loo else 0.0,
        n=int(y.size),
        target_name=target_name,
        notes=notes,
    )


def save_judge(path: str | Path, judge: LearnedJudge) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(judge.to_dict(), indent=2), encoding="utf-8")


def load_judge(path: str | Path) -> LearnedJudge:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return LearnedJudge.from_dict(d)


def redesigned_composite_weights() -> Dict[str, float]:
    """Hand redesign from Phase-2 ablation (drop/downweight anti-correlated intra).

    Ablation: drop_intra → Spearman +0.31; anti_hack alone correlates +0.52.
    Still not Go; use as safer default until learned judge validates.
    """
    return dict(w_rank=0.05, w_sep=0.15, w_hack=0.65, w_intra=0.15)
