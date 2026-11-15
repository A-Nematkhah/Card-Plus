"""Helpers for offline reward re-evaluation and mean-per-step returns (CARD TPE style)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from card_plus.progress import traj_success_flag

RewardFn = Callable[..., Any]


def mean_per_step_return(rewards: Sequence[float]) -> float:
    """Same aggregation as utils.trajectories_return_check: sum(r) / T."""
    arr = np.asarray(list(rewards), dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    return float(np.sum(arr) / float(arr.size))


def extract_component_arrays(traj: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Pull per-step sub-reward arrays stored as ``sub_reward/<name>``."""
    out: Dict[str, np.ndarray] = {}
    for k, v in traj.items():
        if isinstance(k, str) and k.startswith("sub_reward/"):
            name = k[len("sub_reward/") :]
            out[name] = np.asarray(v, dtype=np.float64).ravel()
    return out


def components_from_step_dicts(
    step_dicts: Sequence[Mapping[str, float]],
) -> Dict[str, np.ndarray]:
    keys: List[str] = []
    seen = set()
    for d in step_dicts:
        for k in d.keys():
            if k not in seen:
                seen.add(k)
                keys.append(str(k))
    out: Dict[str, np.ndarray] = {}
    t = len(step_dicts)
    for k in keys:
        out[k] = np.asarray([float(d.get(k, 0.0)) for d in step_dicts], dtype=np.float64)
        assert out[k].shape == (t,)
    return out


def recompute_reward_on_traj(
    reward_fn: RewardFn,
    traj: Dict[str, Any],
    mapping_dicts: Optional[Dict[str, Any]] = None,
    benchmark_name: str = "metaworld",
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Re-run a dense reward on a stored trajectory; return totals + component arrays.

    Meta-World calling convention matches ``utils.trajectories_return_check``:
    ``reward_fn(action=..., obs=..., goal_position=...)``.
    """
    del mapping_dicts  # kept for API symmetry with trajectories_return_check
    obs = traj["obs"]
    n = len(obs)
    totals: List[float] = []
    step_comps: List[Dict[str, float]] = []

    for i in range(n):
        if benchmark_name == "metaworld":
            goal = traj["self.goal_position"][i] if "self.goal_position" in traj else np.zeros(3)
            result = reward_fn(
                action=traj["actions"][i],
                obs=obs[i],
                goal_position=goal,
            )
        elif benchmark_name == "crowdnav":
            pre_obs = obs[i - 1] if i > 0 else obs[i]
            try:
                result = reward_fn(
                    action=traj["actions"][i],
                    obs=obs[i],
                    pre_obs=pre_obs,
                )
            except TypeError:
                # Method-style ``compute_dense_reward(self, action, obs)``:
                # attach pre_obs on a tiny holder when reward_fn is unbound.
                holder = type("PreObsHolder", (), {})()
                holder.pre_obs = pre_obs
                result = reward_fn(holder, traj["actions"][i], obs[i])
        else:
            result = reward_fn(action=traj["actions"][i], obs=obs[i])

        if isinstance(result, tuple) and len(result) == 2:
            total, comps = result
            comps = dict(comps) if comps is not None else {}
        else:
            total, comps = float(result), {}
        totals.append(float(total))
        step_comps.append({str(k): float(v) for k, v in comps.items()})

    return np.asarray(totals, dtype=np.float64), components_from_step_dicts(step_comps)


def sparse_keys_from_pool(
    traj_components: Sequence[Dict[str, np.ndarray]],
    sparse_nonzero_frac: float,
) -> List[str]:
    """A key is sparse if fraction of non-zero steps over the whole pool < threshold."""
    all_keys: List[str] = []
    seen = set()
    for comps in traj_components:
        for k in comps:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    sparse: List[str] = []
    for k in all_keys:
        vals: List[float] = []
        for comps in traj_components:
            if k in comps:
                vals.extend(comps[k].tolist())
        if not vals:
            sparse.append(k)
            continue
        arr = np.asarray(vals, dtype=np.float64)
        nz = float(np.mean(np.abs(arr) > 1e-12))
        if nz < sparse_nonzero_frac:
            sparse.append(k)
    return sparse


def dense_totals(
    totals: np.ndarray,
    components: Dict[str, np.ndarray],
    sparse_keys: Sequence[str],
) -> np.ndarray:
    """Remove sparse components from total (subtract; do not re-sum)."""
    out = np.asarray(totals, dtype=np.float64).copy()
    for k in sparse_keys:
        if k in components:
            out = out - components[k]
    return out


def pool_returns(
    trajs: Sequence[Dict[str, Any]],
    traj_totals: Sequence[np.ndarray],
    traj_components: Sequence[Dict[str, np.ndarray]],
    sparse_keys: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-trajectory mean-per-step returns for full and dense-only rewards."""
    ret_full: List[float] = []
    ret_dense: List[float] = []
    for totals, comps in zip(traj_totals, traj_components):
        ret_full.append(mean_per_step_return(totals))
        dense = dense_totals(totals, comps, sparse_keys)
        ret_dense.append(mean_per_step_return(dense))
    return np.asarray(ret_full, dtype=np.float64), np.asarray(ret_dense, dtype=np.float64)


def load_totals_and_components_from_stored(
    trajs: Sequence[Dict[str, Any]],
) -> Tuple[List[np.ndarray], List[Dict[str, np.ndarray]]]:
    """Use on-policy stored rewards + sub_reward/* when already present."""
    totals_list: List[np.ndarray] = []
    comps_list: List[Dict[str, np.ndarray]] = []
    for traj in trajs:
        totals = np.asarray(traj.get("rewards", []), dtype=np.float64).ravel()
        comps = extract_component_arrays(traj)
        totals_list.append(totals)
        comps_list.append(comps)
    return totals_list, comps_list
