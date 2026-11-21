"""Phase-4 screen ranking predictors from short success/proxy curves."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from card_plus.archive import success_curve_auc
from card_plus.tpe_v2 import spearman_rho


def success_at_k(curve: Sequence[float], k: int = 1) -> float:
    y = list(curve)
    if not y:
        return float("nan")
    k = max(1, min(int(k), len(y)))
    return float(np.mean(y[-k:]))


def curve_slope(curve: Sequence[float], last_m: int = 3) -> float:
    y = np.asarray(list(curve), dtype=np.float64).ravel()
    if y.size == 0:
        return float("nan")
    m = max(2, min(int(last_m), int(y.size)))
    seg = y[-m:]
    x = np.arange(m, dtype=np.float64)
    # Simple least-squares slope
    x = x - x.mean()
    y0 = seg - seg.mean()
    denom = float(np.dot(x, x))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x, y0) / denom)


def proxy_last(proxy_curve: Sequence[float], k: int = 1) -> float:
    return success_at_k(proxy_curve, k=k)


def combined_screen(
    success_curve: Sequence[float],
    proxy_curve: Optional[Sequence[float]] = None,
    *,
    w_succ: float = 0.5,
    w_auc: float = 0.25,
    w_slope: float = 0.15,
    w_proxy: float = 0.10,
) -> float:
    """Weighted mix of short-horizon signals (higher is better)."""
    parts = []
    weights = []
    s = success_at_k(success_curve, k=1)
    if s == s:
        parts.append(s)
        weights.append(float(w_succ))
    auc = success_curve_auc(success_curve)
    if auc == auc:
        parts.append(float(auc))
        weights.append(float(w_auc))
    sl = curve_slope(success_curve, last_m=3)
    if sl == sl:
        # squash slope into roughly [0,1] via sigmoid-ish
        parts.append(float(1.0 / (1.0 + np.exp(-5.0 * sl))))
        weights.append(float(w_slope))
    if proxy_curve:
        p = proxy_last(proxy_curve, k=1)
        if p == p:
            parts.append(float(p))
            weights.append(float(w_proxy))
    if not parts or sum(weights) <= 0:
        return float("nan")
    return float(sum(a * b for a, b in zip(parts, weights)) / sum(weights))


def all_screen_metrics(
    success_curve: Sequence[float],
    proxy_curve: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    return {
        "success_at_1": success_at_k(success_curve, 1),
        "success_at_3": success_at_k(success_curve, 3),
        "curve_auc": float(success_curve_auc(success_curve)),
        "curve_slope_3": curve_slope(success_curve, 3),
        "proxy_at_1": proxy_last(proxy_curve or [], 1)
        if proxy_curve
        else float("nan"),
        "combined": combined_screen(success_curve, proxy_curve),
    }


def spearman_screen_vs_final(
    rows: Sequence[Mapping[str, float]],
    screen_key: str,
    final_key: str = "final_success",
) -> float:
    xs, ys = [], []
    for r in rows:
        x = r.get(screen_key)
        y = r.get(final_key)
        if x is None or y is None:
            continue
        xf, yf = float(x), float(y)
        if xf != xf or yf != yf:
            continue
        xs.append(xf)
        ys.append(yf)
    if len(xs) < 3:
        return 0.0
    return float(spearman_rho(np.asarray(xs), np.asarray(ys)))


def rank_predictors(
    rows: Sequence[Mapping[str, float]],
    metric_keys: Sequence[str],
    final_key: str = "final_success",
) -> List[Dict[str, float]]:
    out = []
    for key in metric_keys:
        sp = spearman_screen_vs_final(rows, key, final_key=final_key)
        out.append({"metric": key, "spearman": sp, "n": float(len(rows))})
    out.sort(key=lambda d: d["spearman"], reverse=True)
    return out
