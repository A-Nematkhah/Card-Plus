"""Phase-3: optimize numeric reward coefficients for offline composite."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from card_plus.dataset.anti_hack_probes import run_anti_hack_probes
from card_plus.dataset.calibration import score_reward_on_episodes
from card_plus.dataset.coeff_ast import coeff_bounds, extract_coeffs, inject_coeffs
from card_plus.dataset.schema import Episode

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


@dataclass
class CoeffOptimizeResult:
    ok: bool
    original_code: str
    tuned_code: str
    original_coeffs: List[float]
    tuned_coeffs: List[float]
    train_composite_before: float
    train_composite_after: float
    val_composite_before: Optional[float] = None
    val_composite_after: Optional[float] = None
    n_evals: int = 0
    method: str = "none"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "original_coeffs": list(self.original_coeffs),
            "tuned_coeffs": list(self.tuned_coeffs),
            "train_composite_before": self.train_composite_before,
            "train_composite_after": self.train_composite_after,
            "val_composite_before": self.val_composite_before,
            "val_composite_after": self.val_composite_after,
            "n_evals": self.n_evals,
            "method": self.method,
            "notes": list(self.notes),
            "tuned_code": self.tuned_code,
        }


def _probe_ok(code: str, *, use_suicide: bool) -> bool:
    from card_plus.dataset.rescore import load_reward_fn_from_code

    fn = load_reward_fn_from_code(code)
    return bool(run_anti_hack_probes(fn, use_suicide_probe=use_suicide).ok)


def _objective_factory(
    base_code: str,
    train_eps: Sequence[Episode],
    *,
    skip_unit: bool,
    use_suicide: bool,
    eval_counter: List[int],
):
    def objective(x: np.ndarray) -> float:
        eval_counter[0] += 1
        code = inject_coeffs(base_code, [float(v) for v in x], skip_unit_literals=skip_unit)
        if not _probe_ok(code, use_suicide=use_suicide):
            return 1e3  # hard constraint: probes must pass
        try:
            composite, _metrics, _ = score_reward_on_episodes(
                code, train_eps, use_suicide_probe=use_suicide
            )
        except Exception as exc:
            logger.debug("coeff objective failed: %s", exc)
            return 1e3
        return -float(composite)  # minimize

    return objective


def _random_search(
    objective,
    bounds: Sequence[Tuple[float, float]],
    x0: Sequence[float],
    *,
    n_iter: int,
    seed: int,
) -> Tuple[np.ndarray, float]:
    rng = np.random.RandomState(int(seed))
    best_x = np.asarray(list(x0), dtype=np.float64)
    best_f = float(objective(best_x))
    for _ in range(max(0, int(n_iter))):
        x = np.asarray(
            [rng.uniform(lo, hi) for lo, hi in bounds],
            dtype=np.float64,
        )
        f = float(objective(x))
        if f < best_f:
            best_f, best_x = f, x
    return best_x, best_f


def _differential_evolution(
    objective,
    bounds: Sequence[Tuple[float, float]],
    *,
    maxiter: int,
    seed: int,
) -> Optional[Tuple[np.ndarray, float, str]]:
    try:
        from scipy.optimize import differential_evolution
    except Exception:
        return None
    result = differential_evolution(
        objective,
        bounds=list(bounds),
        maxiter=int(maxiter),
        seed=int(seed),
        polish=False,
        updating="immediate",
        workers=1,
    )
    return (
        np.asarray(result.x, dtype=np.float64),
        float(result.fun),
        "differential_evolution",
    )


def _cma_es(
    objective,
    bounds: Sequence[Tuple[float, float]],
    x0: Sequence[float],
    *,
    maxiter: int,
    seed: int,
) -> Optional[Tuple[np.ndarray, float, str]]:
    try:
        import cma  # type: ignore
    except Exception:
        return None
    lo = np.asarray([b[0] for b in bounds], dtype=np.float64)
    hi = np.asarray([b[1] for b in bounds], dtype=np.float64)
    sigma0 = float(np.mean(hi - lo) / 4.0) or 0.1
    opts = {
        "bounds": [lo.tolist(), hi.tolist()],
        "maxiter": int(maxiter),
        "seed": int(seed),
        "verbose": -9,
    }
    es = cma.CMAEvolutionStrategy(list(x0), sigma0, opts)
    while not es.stop():
        xs = es.ask()
        fs = [objective(np.asarray(x, dtype=np.float64)) for x in xs]
        es.tell(xs, fs)
    xbest = np.asarray(es.result.xbest, dtype=np.float64)
    fbest = float(es.result.fbest)
    return xbest, fbest, "cma_es"


def optimize_reward_coefficients(
    code: str,
    train_episodes: Sequence[Episode],
    *,
    val_episodes: Optional[Sequence[Episode]] = None,
    method: str = "auto",  # auto | de | cma | random
    max_evals: int = 40,
    seed: int = 0,
    skip_unit_literals: bool = True,
    use_suicide_probe: bool = False,
    scale_lo: float = 0.1,
    scale_hi: float = 10.0,
) -> CoeffOptimizeResult:
    """Tune numeric literals to maximize train composite; report val composites."""
    notes: List[str] = []
    coeffs, _sites = extract_coeffs(code, skip_unit_literals=skip_unit_literals)
    if not coeffs:
        notes.append("no tunable numeric literals found")
        comp, _, _ = score_reward_on_episodes(
            code, train_episodes, use_suicide_probe=use_suicide_probe
        )
        return CoeffOptimizeResult(
            ok=False,
            original_code=code,
            tuned_code=code,
            original_coeffs=[],
            tuned_coeffs=[],
            train_composite_before=float(comp),
            train_composite_after=float(comp),
            notes=notes,
            method="none",
        )

    before_train, _, _ = score_reward_on_episodes(
        code, train_episodes, use_suicide_probe=use_suicide_probe
    )
    before_val = None
    if val_episodes is not None:
        before_val, _, _ = score_reward_on_episodes(
            code, val_episodes, use_suicide_probe=use_suicide_probe
        )

    bounds = coeff_bounds(coeffs, scale_lo=scale_lo, scale_hi=scale_hi)
    eval_counter = [0]
    objective = _objective_factory(
        code,
        train_episodes,
        skip_unit=skip_unit_literals,
        use_suicide=use_suicide_probe,
        eval_counter=eval_counter,
    )

    method_key = str(method or "auto").lower()
    best_x = np.asarray(coeffs, dtype=np.float64)
    best_f = float(objective(best_x))
    used = "random"
    maxiter = max(2, int(max_evals) // max(1, len(coeffs)))

    if method_key in ("auto", "cma"):
        cma_out = _cma_es(
            objective, bounds, coeffs, maxiter=maxiter, seed=seed
        )
        if cma_out is not None:
            best_x, best_f, used = cma_out
        elif method_key == "cma":
            notes.append("cma package missing; falling back")
            method_key = "auto"

    if used == "random" and method_key in ("auto", "de"):
        de_out = _differential_evolution(
            objective, bounds, maxiter=maxiter, seed=seed
        )
        if de_out is not None:
            best_x, best_f, used = de_out
        elif method_key == "de":
            notes.append("scipy differential_evolution unavailable; using random")

    if used == "random" or method_key == "random":
        n_rand = max(int(max_evals) - eval_counter[0], 8)
        best_x, best_f = _random_search(
            objective, bounds, coeffs, n_iter=n_rand, seed=seed
        )
        used = "random"

    tuned_code = inject_coeffs(
        code, [float(v) for v in best_x], skip_unit_literals=skip_unit_literals
    )
    after_train, _, _ = score_reward_on_episodes(
        tuned_code, train_episodes, use_suicide_probe=use_suicide_probe
    )
    after_val = None
    if val_episodes is not None:
        after_val, _, _ = score_reward_on_episodes(
            tuned_code, val_episodes, use_suicide_probe=use_suicide_probe
        )

    ok = bool(after_train >= before_train - 1e-9)
    if val_episodes is not None and before_val is not None and after_val is not None:
        if after_val + 1e-9 < before_val:
            notes.append(
                f"val composite dropped ({before_val:.4f} → {after_val:.4f}); "
                "keeping tuned code but flagging ok=False for exit criterion"
            )
            ok = False
        else:
            notes.append(
                f"val composite {before_val:.4f} → {after_val:.4f} (non-decreasing)"
            )

    return CoeffOptimizeResult(
        ok=ok,
        original_code=code,
        tuned_code=tuned_code,
        original_coeffs=[float(v) for v in coeffs],
        tuned_coeffs=[float(v) for v in best_x],
        train_composite_before=float(before_train),
        train_composite_after=float(after_train),
        val_composite_before=None if before_val is None else float(before_val),
        val_composite_after=None if after_val is None else float(after_val),
        n_evals=int(eval_counter[0]),
        method=used,
        notes=notes,
    )


def write_coeff_optimize_report(result: CoeffOptimizeResult, path: PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return p
