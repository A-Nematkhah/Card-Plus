"""Shared statistics helpers for CARD+ reporting (Wilson CI, bootstrap)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class WilsonCI:
    mean: float
    low: float
    high: float
    n: int
    z: float


def wilson_ci(successes: int, n: int, z: float = 1.96) -> WilsonCI:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return WilsonCI(mean=float("nan"), low=float("nan"), high=float("nan"), n=0, z=z)
    successes = int(np.clip(successes, 0, n))
    p = successes / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * float(
        np.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    )
    return WilsonCI(
        mean=float(p),
        low=float(max(0.0, center - margin)),
        high=float(min(1.0, center + margin)),
        n=int(n),
        z=float(z),
    )


def bootstrap_summary(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap mean/median with percentile CI. Empty input → NaNs."""
    arr = np.asarray(list(values), dtype=np.float64).ravel()
    if arr.size == 0:
        return {
            "n": 0.0,
            "mean": float("nan"),
            "median": float("nan"),
            "mean_low": float("nan"),
            "mean_high": float("nan"),
            "median_low": float("nan"),
            "median_high": float("nan"),
        }
    rng = np.random.RandomState(int(seed))
    idx = rng.randint(0, arr.size, size=(int(n_boot), arr.size))
    samples = arr[idx]
    means = samples.mean(axis=1)
    medians = np.median(samples, axis=1)
    lo = 100.0 * (alpha / 2.0)
    hi = 100.0 * (1.0 - alpha / 2.0)
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "mean_low": float(np.percentile(means, lo)),
        "mean_high": float(np.percentile(means, hi)),
        "median_low": float(np.percentile(medians, lo)),
        "median_high": float(np.percentile(medians, hi)),
    }


def per_seed_report(
    seed_to_successes: Dict[int, Sequence[float]],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> Dict[str, object]:
    """Per-seed means + pooled Wilson (treating each value as Bernoulli if in {0,1})."""
    per_seed: Dict[str, Dict[str, float]] = {}
    pooled: List[float] = []
    for s, vals in sorted(seed_to_successes.items(), key=lambda kv: int(kv[0])):
        arr = [float(v) for v in vals]
        pooled.extend(arr)
        summary = bootstrap_summary(arr, n_boot=n_boot, seed=seed + int(s))
        per_seed[str(s)] = summary
    if pooled and all(v in (0.0, 1.0) for v in pooled):
        w = wilson_ci(int(round(sum(pooled))), len(pooled))
        pooled_wilson = {
            "mean": w.mean,
            "low": w.low,
            "high": w.high,
            "n": float(w.n),
        }
    else:
        pooled_wilson = bootstrap_summary(pooled, n_boot=n_boot, seed=seed)
    return {"per_seed": per_seed, "pooled": pooled_wilson}


def format_ci(mean: float, low: float, high: float, digits: int = 3) -> str:
    if any(x != x for x in (mean, low, high)):  # NaN check
        return "nan"
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"
