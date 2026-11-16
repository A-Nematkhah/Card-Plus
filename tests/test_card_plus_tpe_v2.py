"""TPE v2 unit tests (no RL / no LLM)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from card_plus.config import CardPlusConfig
from card_plus.tpe_v2 import run_tpe_v2, should_skip_rl_for_tpe_v2


def _make_traj(
    *,
    length: int,
    success: bool,
    obj_start: float,
    obj_end: float,
    goal: float = 1.0,
    rewards: Optional[np.ndarray] = None,
    components: Optional[Dict[str, np.ndarray]] = None,
) -> dict:
    t = length
    obs = np.zeros((t, 39), dtype=np.float64)
    # layout: hand 0:3, gripper 3, obj 4:7
    xs = np.linspace(obj_start, obj_end, t)
    obs[:, 4] = xs
    goal_pos = np.tile(np.array([goal, 0.0, 0.0]), (t, 1))
    rew = rewards if rewards is not None else np.zeros(t)
    traj = {
        "obs": obs,
        "actions": np.zeros((t, 4)),
        "rewards": rew,
        "dones": np.zeros(t, dtype=bool),
        "infos": [{"success": success}] * (t - 1) + [{"success": success, "is_success": success}],
        "self.obj1.position": np.stack([xs, np.zeros(t), np.zeros(t)], axis=1),
        "self.goal_position": goal_pos,
    }
    if components:
        for k, arr in components.items():
            traj[f"sub_reward/{k}"] = np.asarray(arr, dtype=np.float64)
    return traj


def test_sparse_bonus_hacks_v1_pass_but_v2_hacked():
    """Only a big success_reward orders trajs → v1-style pass inputs, hacked=True."""
    trajs = []
    # 8 success with high sparse terminal bonus; progress also high
    for i in range(8):
        t = 20
        dense = np.full(t, 0.01)  # weak dense
        sparse = np.zeros(t)
        sparse[-1] = 10.0
        total = dense + sparse
        trajs.append(
            _make_traj(
                length=t,
                success=True,
                obj_start=0.0,
                obj_end=0.95,
                rewards=total,
                components={"approach": dense, "success_reward": sparse},
            )
        )
    # 8 failures: no sparse bonus; slightly different dense; low progress
    for i in range(8):
        t = 20
        dense = np.full(t, 0.02)
        sparse = np.zeros(t)
        total = dense + sparse
        trajs.append(
            _make_traj(
                length=t,
                success=False,
                obj_start=0.0,
                obj_end=0.1,
                rewards=total,
                components={"approach": dense, "success_reward": sparse},
            )
        )

    cfg = CardPlusConfig(
        use_tpe_v2=True,
        tpe_v2_min_trajs=10,
        tpe_v2_tau_dense=0.3,
        sparse_nonzero_frac=0.05,
    )
    # Simulate v1 pass
    res = run_tpe_v2(
        trajs,
        pass_v1=True,
        acc_v1=1.0,
        cfg=cfg,
        task_name="drawer-open-v2",
        reward_fn=None,
        use_stored_components=True,
    )
    assert res.hacked is True
    assert res.rho_dense < cfg.tpe_v2_tau_dense
    assert "success_reward" in res.sparse_keys
    assert "Dense-Shaping Check" in res.feedback_text


def test_good_dense_shaping_not_hacked():
    trajs = []
    for progress_end, success in [(0.95, True), (0.9, True), (0.85, True), (0.8, True),
                                    (0.75, True), (0.2, False), (0.15, False), (0.1, False),
                                    (0.05, False), (0.0, False)]:
        t = 30
        xs = np.linspace(0.0, progress_end, t)
        # dense reward tracks progress smoothly
        dense = xs * 2.0
        sparse = np.zeros(t)
        if success:
            sparse[-1] = 0.5  # small non-dominant sparse
        total = dense + sparse
        trajs.append(
            _make_traj(
                length=t,
                success=success,
                obj_start=0.0,
                obj_end=progress_end,
                rewards=total,
                components={"approach": dense, "success_reward": sparse},
            )
        )
    # pad to >= 10
    while len(trajs) < 10:
        trajs.append(trajs[-1])

    cfg = CardPlusConfig(use_tpe_v2=True, tpe_v2_min_trajs=10, tpe_v2_tau_dense=0.3, sparse_nonzero_frac=0.05)
    res = run_tpe_v2(
        trajs, pass_v1=True, acc_v1=1.0, cfg=cfg, task_name="drawer-open-v2", use_stored_components=True
    )
    assert res.hacked is False
    assert res.rho_dense >= cfg.tpe_v2_tau_dense
    assert res.feedback_text == ""


def test_degenerate_and_small_pool_fallback():
    cfg = CardPlusConfig(use_tpe_v2=True, tpe_v2_min_trajs=10)
    # too few trajs
    trajs = [
        _make_traj(length=5, success=True, obj_start=0, obj_end=1, rewards=np.ones(5),
                   components={"a": np.ones(5)}),
        _make_traj(length=5, success=False, obj_start=0, obj_end=0.1, rewards=np.zeros(5),
                   components={"a": np.zeros(5)}),
    ]
    res = run_tpe_v2(trajs, pass_v1=True, acc_v1=1.0, cfg=cfg, task_name="drawer-open-v2")
    assert res.skipped is True
    assert res.verdict == "fallback_v1"
    assert should_skip_rl_for_tpe_v2(cfg, res) is False

    # constant returns / progress — no crash
    big = []
    for i in range(12):
        big.append(
            _make_traj(
                length=8,
                success=(i % 2 == 0),
                obj_start=0.5,
                obj_end=0.5,
                rewards=np.full(8, 1.0),
                components={"c": np.full(8, 1.0)},
            )
        )
    res2 = run_tpe_v2(big, pass_v1=True, acc_v1=0.9, cfg=cfg, task_name="generic")
    assert res2.rho_full == 0.0 or isinstance(res2.rho_full, float)
    assert not np.isnan(res2.rho_full)


def test_no_success_pool_overrides_pass_v1_and_computes_hacked():
    cfg = CardPlusConfig(use_tpe_v2=True, tpe_v2_min_trajs=10, tpe_v2_tau_dense=0.3, tpe_v2_mode="gate")
    trajs = []
    for i in range(12):
        end = 0.9 if i < 6 else 0.1  # progress varies; all failed
        t = 16
        dense = np.linspace(0.0, end, t)
        trajs.append(
            _make_traj(
                length=t,
                success=False,
                obj_start=0.0,
                obj_end=end,
                rewards=dense,
                components={"approach": dense},
            )
        )
    res = run_tpe_v2(trajs, pass_v1=True, acc_v1=1.0, cfg=cfg, task_name="drawer-open-v2")
    assert res.no_success_pool is True
    assert res.pass_v1_overridden is True
    # rho_full should be high (dense tracks progress) → pass_v1 True; rho_dense same → not hacked
    assert res.pass_v1 is True
    assert res.hacked is False
    assert should_skip_rl_for_tpe_v2(cfg, res) is False  # skip-RL still only gate AND hacked


def test_gate_mode_skips_rl_when_hacked():
    cfg = CardPlusConfig(use_tpe_v2=True, tpe_v2_mode="gate")
    from card_plus.tpe_v2 import TPEv2Result

    hacked = TPEv2Result(pass_v1=True, acc_v1=1.0, rho_full=0.9, rho_dense=0.0, hacked=True, verdict="hacked")
    assert should_skip_rl_for_tpe_v2(cfg, hacked) is True
    cfg_warn = CardPlusConfig(use_tpe_v2=True, tpe_v2_mode="warn")
    assert should_skip_rl_for_tpe_v2(cfg_warn, hacked) is False


def test_both_rhos_weak_not_hacked():
    """Distance-penalty style: v1 can pass while both rhos are weak → not hacked."""
    from card_plus.tpe_v2 import is_dense_shaping_hacked

    cfg = CardPlusConfig(use_tpe_v2=True, tpe_v2_tau_dense=0.3, tpe_v2_sparse_gap=0.2)
    assert (
        is_dense_shaping_hacked(
            pass_v1=True,
            rho_full=-0.38,
            rho_dense=-0.20,
            sparse_keys=["success_reward", "collision_reward"],
            cfg=cfg,
        )
        is False
    )
    # Sparse clearly carries ordering: full high, dense low, gap large → hacked
    assert (
        is_dense_shaping_hacked(
            pass_v1=True,
            rho_full=0.85,
            rho_dense=-0.10,
            sparse_keys=["success_reward"],
            cfg=cfg,
        )
        is True
    )
