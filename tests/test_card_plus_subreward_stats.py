"""Sub-reward statistics feedback tests."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from card_plus.config import CardPlusConfig
from card_plus.subreward_stats import format_subreward_stats_feedback


def _traj(
    success: bool,
    components: Dict[str, np.ndarray],
    obj_end: float = 0.5,
) -> dict:
    t = len(next(iter(components.values())))
    xs = np.linspace(0.0, obj_end, t)
    obs = np.zeros((t, 39))
    obs[:, 4] = xs
    total = sum(components.values())
    out = {
        "obs": obs,
        "actions": np.zeros((t, 4)),
        "rewards": total,
        "dones": np.zeros(t, dtype=bool),
        "infos": [{"success": success}] * t,
        "self.obj1.position": np.stack([xs, np.zeros(t), np.zeros(t)], axis=1),
        "self.goal_position": np.tile([1.0, 0.0, 0.0], (t, 1)),
    }
    for k, v in components.items():
        out[f"sub_reward/{k}"] = v
    return out


def test_each_flag_fires_and_control_clean():
    t = 40
    # DOMINANT + SPARSE success bonus
    dominant = np.full(t, 10.0)
    sparse = np.zeros(t)
    sparse[-1] = 1.0
    # HIGHER_IN_FAILURES: bigger on fail trajs
    bad = np.full(t, 1.0)
    # NEG_CORR: decreases as progress increases — put high values on low-progress fails
    # INERT
    inert = np.full(t, 0.0)

    succ = _traj(True, {"dom": dominant * 0.1, "sparse_b": sparse, "inert_z": inert, "ok": np.linspace(0, 1, t)}, obj_end=0.9)
    fail = _traj(True, {"dom": dominant, "sparse_b": sparse * 0, "inert_z": inert, "ok": np.linspace(0, 0.1, t)}, obj_end=0.1)
    # make fail actually failed
    fail["infos"] = [{"success": False}] * t
    # higher bad component on failures
    fail[f"sub_reward/bad"] = bad * 5.0
    succ[f"sub_reward/bad"] = bad * 0.1

    trajs = [succ, fail] * 5
    cfg = CardPlusConfig(
        use_subreward_stats=True,
        dominant_share=0.5,
        sparse_nonzero_frac=0.05,
        inert_std_eps=1e-6,
        neg_corr_threshold=-0.2,
        max_feedback_lines=40,
    )
    text = format_subreward_stats_feedback(trajs, cfg, task_name="drawer-open-v2")
    assert "# Sub-reward Statistics" in text
    assert "[DOMINANT]" in text or "dom" in text
    assert "[SPARSE]" in text or "sparse_b" in text
    assert "[INERT]" in text
    assert "[HIGHER_IN_FAILURES]" in text

    # Control: balanced shaping — no DOMINANT / INERT / SPARSE
    good_a = np.linspace(0.0, 1.0, t)
    good_b = np.linspace(0.0, 1.0, t)  # equal |mass| → neither DOMINANT
    g_succ = _traj(True, {"a": good_a, "b": good_b}, obj_end=0.9)
    g_fail = _traj(False, {"a": good_a * 0.2, "b": good_b * 0.2}, obj_end=0.1)
    control = format_subreward_stats_feedback([g_succ, g_fail] * 5, cfg, task_name="drawer-open-v2")
    assert "[DOMINANT]" not in control
    assert "[INERT]" not in control


def test_line_cap_respected():
    t = 20
    comps = {f"r{i}": np.full(t, float(i + 1)) for i in range(30)}
    # make them sparse/inert variously
    for i in range(30):
        if i % 2 == 0:
            comps[f"r{i}"] = np.zeros(t)
    traj = _traj(True, comps, obj_end=0.5)
    traj_f = _traj(False, comps, obj_end=0.1)
    cfg = CardPlusConfig(use_subreward_stats=True, max_feedback_lines=12, dominant_share=0.01)
    text = format_subreward_stats_feedback([traj, traj_f], cfg, task_name="generic")
    assert len(text.splitlines()) <= 12
