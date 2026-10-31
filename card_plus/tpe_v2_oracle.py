#!/usr/bin/env python
"""Run TPE v2 on stored trajectories with two rewards: MW-style oracle vs success-bonus-only.

Prints rho_full, rho_dense, hacked for each. No RL / no LLM.

    python -m card_plus.tpe_v2_oracle
    python -m card_plus.tpe_v2_oracle --traj_pkl PATH.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from card_plus.config import CardPlusConfig
from card_plus.tpe_v2 import run_tpe_v2


def _tolerance(x: float, bounds: Tuple[float, float], margin: float) -> float:
    """Meta-World long_tail-style tolerance (self-contained; no mujoco import)."""
    lower, upper = bounds
    if lower <= x <= upper:
        return 1.0
    d = lower - x if x < lower else x - upper
    if margin <= 0:
        return 0.0
    z = d / margin
    return float(1.0 / (1.0 + z + 0.5 * z * z))


def metaworld_drawer_open_oracle(
    action: Any,
    obs: np.ndarray,
    goal_position: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """Obs-only clone of SawyerDrawerOpenEnvV2.compute_reward terms.

    Uses stored goal as target handle position. ``maxDist`` default 0.16 matches
    the env's closed-to-open offset. Never reads a ground-truth env reward field.
    """
    del action
    obs = np.asarray(obs, dtype=np.float64).ravel()
    goal = np.asarray(goal_position, dtype=np.float64).ravel()[:3]
    gripper = obs[:3]
    handle = obs[4:7]
    handle_error = float(np.linalg.norm(handle - goal))
    opening = _tolerance(handle_error, (0.0, 0.02), margin=0.16)
    scale = np.array([3.0, 3.0, 1.0])
    gripper_error = float(np.linalg.norm((handle - gripper) * scale))
    caging = _tolerance(gripper_error, (0.0, 0.01), margin=0.5)
    caging_v = float(5.0 * caging)
    opening_v = float(5.0 * opening)
    total = caging_v + opening_v
    return total, {"caging": caging_v, "opening": opening_v}


def success_bonus_only(
    action: Any,
    obs: np.ndarray,
    goal_position: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """Sparse success bonus: 10 iff handle is within 5 cm of the goal, else 0."""
    del action
    obs = np.asarray(obs, dtype=np.float64).ravel()
    goal = np.asarray(goal_position, dtype=np.float64).ravel()[:3]
    handle = obs[4:7]
    # Tight threshold so the bonus is sparse over the pool (last step of near-goal trajs).
    hit = float(np.linalg.norm(handle - goal) < 0.005)
    bonus = 10.0 * hit
    return bonus, {"success_reward": bonus}


def _synthetic_pool(n: int = 16) -> List[dict]:
    trajs: List[dict] = []
    for i in range(n):
        success = i < n // 2
        t = 20
        end = 0.02 if success else 0.20
        xs = np.linspace(0.20, end, t)
        obs = np.zeros((t, 39), dtype=np.float64)
        obs[:, 4] = 0.0
        obs[:, 5] = xs
        obs[:, 6] = 0.09
        goal = np.tile(np.array([0.0, 0.02, 0.09]), (t, 1))
        trajs.append(
            {
                "obs": obs,
                "actions": np.zeros((t, 4)),
                "rewards": np.zeros(t),
                "dones": np.zeros(t, dtype=bool),
                "infos": [{"success": success, "is_success": success}] * t,
                "self.obj1.position": obs[:, 4:7],
                "self.goal_position": goal,
            }
        )
    return trajs


def _load_trajs(path: str) -> Sequence[dict]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "trajectories_for_check_list" in obj:
        return list(obj["trajectories_for_check_list"])
    if isinstance(obj, list):
        return obj
    raise TypeError(f"Unsupported pickle payload: {type(obj)}")


def _report(name: str, result: Any) -> None:
    print(
        f"{name}: rho_full={result.rho_full:.4f} rho_dense={result.rho_dense:.4f} "
        f"hacked={result.hacked} sparse_keys={result.sparse_keys} verdict={result.verdict}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="TPE v2 oracle vs success-bonus-only")
    p.add_argument("--traj_pkl", type=str, default=None)
    p.add_argument("--task_name", type=str, default="drawer-open-v2")
    args = p.parse_args()
    trajs = _load_trajs(args.traj_pkl) if args.traj_pkl else _synthetic_pool()
    print(f"pool size={len(trajs)} source={'pkl' if args.traj_pkl else 'synthetic'}")
    cfg = CardPlusConfig(use_tpe_v2=True, tpe_v2_min_trajs=1)
    for name, fn in (
        ("oracle", metaworld_drawer_open_oracle),
        ("success_bonus_only", success_bonus_only),
    ):
        res = run_tpe_v2(
            trajs,
            pass_v1=True,
            acc_v1=1.0,
            cfg=cfg,
            task_name=args.task_name,
            reward_fn=fn,
            benchmark_name="metaworld",
        )
        _report(name, res)


if __name__ == "__main__":
    main()
