"""TPE v2: multi-level ordering + dense-shaping / reward-hacking check."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from card_plus.config import CardPlusConfig
from card_plus.progress import (
    compute_progress_values,
    traj_success_flag,
    validate_progress_fn,
)
from card_plus.reward_utils import (
    load_totals_and_components_from_stored,
    pool_returns,
    recompute_reward_on_traj,
    sparse_keys_from_pool,
)

logger = logging.getLogger(__name__)

DENSE_SHAPING_FEEDBACK_TEMPLATE = """# Dense-Shaping Check
Rank correlation between per-step return and task progress: full reward rho={rho_full:.2f}; without sparse components ({sparse_keys}) rho={rho_dense:.2f}.
The ordering of trajectories is mostly produced by the sparse component(s), so the dense terms do not yet guide the agent toward task progress. Strengthen the dense shaping (e.g. terms that increase smoothly with progress) instead of only enlarging the sparse bonus.
"""


@dataclass
class TPEv2Result:
    pass_v1: bool
    acc_v1: float
    rho_full: float
    rho_dense: float
    sparse_keys: List[str] = field(default_factory=list)
    hacked: bool = False
    verdict: str = "skip"
    feedback_text: str = ""
    skipped: bool = False
    no_success_pool: bool = False
    pass_v1_overridden: bool = False


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation; constant / NaN inputs → 0.0 with a log."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or y.size < 2 or x.size != y.size:
        logger.info("spearman_rho: insufficient / mismatched length → 0.0")
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        logger.info("spearman_rho: constant input → 0.0")
        return 0.0
    try:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(x, y)
        if rho is None or np.isnan(rho):
            logger.info("spearman_rho: NaN from scipy → 0.0")
            return 0.0
        return float(rho)
    except Exception:
        # Numpy fallback: Pearson of ranks
        rx = _rankdata(x)
        ry = _rankdata(y)
        rx = rx - rx.mean()
        ry = ry - ry.mean()
        denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
        if denom < 1e-12:
            logger.info("spearman_rho: zero variance after ranking → 0.0")
            return 0.0
        return float(np.sum(rx * ry) / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty_like(a, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    # average ties
    sorted_a = a[order]
    i = 0
    while i < len(sorted_a):
        j = i
        while j < len(sorted_a) and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j - 1]])
            ranks[order[i:j]] = avg
        i = j
    return ranks


def format_dense_shaping_feedback(
    rho_full: float,
    rho_dense: float,
    sparse_keys: Sequence[str],
) -> str:
    keys_str = ", ".join(sparse_keys) if sparse_keys else "none"
    return DENSE_SHAPING_FEEDBACK_TEMPLATE.format(
        rho_full=rho_full,
        rho_dense=rho_dense,
        sparse_keys=keys_str,
    )


def run_tpe_v2(
    trajs: Sequence[Dict[str, Any]],
    *,
    pass_v1: bool,
    acc_v1: float,
    cfg: CardPlusConfig,
    task_name: Optional[str] = None,
    reward_fn: Optional[Callable[..., Any]] = None,
    mapping_dicts: Optional[Dict[str, Any]] = None,
    use_stored_components: bool = True,
    benchmark_name: str = "metaworld",
) -> TPEv2Result:
    """Run TPE v2 on a trajectory pool.

    If ``reward_fn`` is given, recompute totals/components offline.
    Else use stored ``rewards`` / ``sub_reward/*`` (post-training rollouts).
    """
    n = len(trajs)
    if n < cfg.tpe_v2_min_trajs:
        logger.info(
            "TPE v2 skipped: pool has %d < tpe_v2_min_trajs=%d; falling back to v1.",
            n,
            cfg.tpe_v2_min_trajs,
        )
        return TPEv2Result(
            pass_v1=pass_v1,
            acc_v1=acc_v1,
            rho_full=0.0,
            rho_dense=0.0,
            sparse_keys=[],
            hacked=False,
            verdict="fallback_v1",
            skipped=True,
        )

    validate_progress_fn(
        trajs,
        task_name=task_name,
        auc_min=cfg.progress_validation_auc_min,
    )
    progress = compute_progress_values(trajs, task_name=task_name)

    if reward_fn is not None:
        totals_list: List[np.ndarray] = []
        comps_list: List[Dict[str, np.ndarray]] = []
        for traj in trajs:
            totals, comps = recompute_reward_on_traj(
                reward_fn,
                traj,
                mapping_dicts=mapping_dicts,
                benchmark_name=benchmark_name,
            )
            totals_list.append(totals)
            comps_list.append(comps)
    elif use_stored_components:
        totals_list, comps_list = load_totals_and_components_from_stored(trajs)
    else:
        totals_list, comps_list = load_totals_and_components_from_stored(trajs)

    sparse_keys = sparse_keys_from_pool(comps_list, cfg.sparse_nonzero_frac)
    ret_full, ret_dense = pool_returns(trajs, totals_list, comps_list, sparse_keys)

    rho_full = spearman_rho(ret_full, progress)
    rho_dense = spearman_rho(ret_dense, progress)

    n_succ = sum(1 for t in trajs if traj_success_flag(t))
    no_success_pool = n_succ == 0
    pass_v1_overridden = False
    if no_success_pool:
        original_pass = pass_v1
        pass_v1 = bool(rho_full >= cfg.tpe_v2_tau_dense)
        pass_v1_overridden = True
        logger.info(
            "TPE v2 no-success pool: overriding pass_v1 %s -> %s "
            "(rho_full=%.3f, tau=%.3f). Skip-RL still uses gate AND hacked only.",
            original_pass,
            pass_v1,
            rho_full,
            cfg.tpe_v2_tau_dense,
        )
    hacked = bool(pass_v1 and rho_dense < cfg.tpe_v2_tau_dense)
    if no_success_pool:
        verdict = "hacked_no_success_pool" if hacked else (
            "pass_no_success_pool" if pass_v1 else "fail_no_success_pool"
        )
    elif hacked:
        verdict = "hacked"
    elif pass_v1:
        verdict = "pass"
    else:
        verdict = "fail_v1"

    feedback = ""
    if hacked:
        feedback = format_dense_shaping_feedback(rho_full, rho_dense, sparse_keys)

    return TPEv2Result(
        pass_v1=pass_v1,
        acc_v1=acc_v1,
        rho_full=rho_full,
        rho_dense=rho_dense,
        sparse_keys=list(sparse_keys),
        hacked=hacked,
        verdict=verdict,
        feedback_text=feedback,
        skipped=False,
        no_success_pool=no_success_pool,
        pass_v1_overridden=pass_v1_overridden,
    )


def should_skip_rl_for_tpe_v2(cfg: CardPlusConfig, result: TPEv2Result) -> bool:
    """Gate mode: treat hacked as TPE failure → skip RL."""
    if not cfg.use_tpe_v2:
        return False
    if result.skipped:
        return False
    return cfg.tpe_v2_mode == "gate" and result.hacked
