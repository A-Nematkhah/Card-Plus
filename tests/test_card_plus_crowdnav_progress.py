"""CrowdNav progress measure (obs layout, no GT reward)."""

from __future__ import annotations

import numpy as np

from card_plus.progress import crowdnav_goal_progress, get_progress_fn
from card_plus.tpe_v2 import run_tpe_v2
from card_plus.config import CardPlusConfig


def _crowdnav_traj(*, success: bool, start_xy, end_xy, goal_xy, length: int = 16) -> dict:
    xs = np.linspace(start_xy[0], end_xy[0], length)
    ys = np.linspace(start_xy[1], end_xy[1], length)
    obs = np.zeros((length, 16 + 5 * 20), dtype=np.float64)
    obs[:, 0] = xs
    obs[:, 1] = ys
    obs[:, 4] = goal_xy[0]
    obs[:, 5] = goal_xy[1]
    pos = np.stack([xs, ys], axis=1)
    goal = np.tile(np.asarray(goal_xy, dtype=np.float64), (length, 1))
    return {
        "obs": obs,
        "actions": np.zeros((length, 2)),
        "rewards": np.zeros(length),
        "dones": np.zeros(length, dtype=bool),
        "infos": [{"success": float(success), "is_success": success}] * length,
        "self.robot.position": pos,
        "self.robot.goal_position": goal,
        "sub_reward/approach": np.linspace(0.0, 1.0 if success else 0.1, length),
        "sub_reward/success_reward": np.array([0.0] * (length - 1) + ([10.0] if success else [0.0])),
    }


def test_progress_fn_registered_for_circle_crossing():
    assert get_progress_fn("circle-crossing-v0") is crowdnav_goal_progress


def test_closer_to_goal_has_higher_progress():
    near = _crowdnav_traj(success=True, start_xy=(4.0, 0.0), end_xy=(0.2, 0.0), goal_xy=(0.0, 0.0))
    far = _crowdnav_traj(success=False, start_xy=(4.0, 0.0), end_xy=(3.5, 0.0), goal_xy=(0.0, 0.0))
    assert crowdnav_goal_progress(near) > crowdnav_goal_progress(far)
    assert 0.0 <= crowdnav_goal_progress(far) <= 1.0
    assert crowdnav_goal_progress(near) > 0.8


def test_tpe_v2_crowdnav_sparse_hack_detected():
    trajs = []
    for i in range(6):
        t = _crowdnav_traj(success=True, start_xy=(4.0, 0.0), end_xy=(0.2, 0.0), goal_xy=(0.0, 0.0))
        # Weak constant dense term: only the sparse terminal bonus orders success vs fail.
        t["sub_reward/approach"] = np.full(len(t["rewards"]), 0.01)
        t["rewards"] = t["sub_reward/approach"] + t["sub_reward/success_reward"]
        trajs.append(t)
    for i in range(6):
        t = _crowdnav_traj(success=False, start_xy=(4.0, 0.0), end_xy=(3.5, 0.0), goal_xy=(0.0, 0.0))
        t["sub_reward/approach"] = np.full(len(t["rewards"]), 0.02)
        t["rewards"] = t["sub_reward/approach"] + t["sub_reward/success_reward"]
        trajs.append(t)
    cfg = CardPlusConfig(use_tpe_v2=True, tpe_v2_min_trajs=10, tpe_v2_tau_dense=0.3, sparse_nonzero_frac=0.05)
    res = run_tpe_v2(
        trajs,
        pass_v1=True,
        acc_v1=1.0,
        cfg=cfg,
        task_name="circle-crossing-v0",
        benchmark_name="crowdnav",
        use_stored_components=True,
    )
    assert res.hacked is True
    assert "success_reward" in res.sparse_keys
