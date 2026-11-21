"""Phase-3 coefficient AST + optimize + shaping helpers."""

from __future__ import annotations

import numpy as np

from card_plus.dataset.coeff_ast import extract_coeffs, inject_coeffs
from card_plus.dataset.coeff_optimize import optimize_reward_coefficients
from card_plus.dataset.reward_shaping import (
    RunningScaleNormalizer,
    STRUCTURE_ONLY_PROMPT,
    potential_based_progress_value,
)
from card_plus.dataset.schema import AgentState, Episode, EpisodeLabels, ScenarioSpec, StepRecord

REWARD = """
def compute_dense_reward(self, action, obs):
    import numpy as np
    progress = -0.2 * float(np.linalg.norm(obs[0:2] - obs[4:6]))
    stall = -0.05
    collision = -40.0 if obs[7] < 0.0 else 0.0
    success = 25.0 if float(np.linalg.norm(obs[0:2] - obs[4:6])) <= obs[6] else 0.0
    total = progress + stall + collision + success
    return total, {"progress": progress, "stall": stall, "collision": collision, "success": success}
"""


def _ep(sid: str, outcome: str, near: float = 1.0) -> Episode:
    robot = AgentState(0, 0.0, 0.0, 0.0, 0.0, 0.3, 5.0, 0.0, True)
    steps = []
    n = 3
    for i in range(n):
        obs = np.zeros(116, dtype=np.float64)
        if outcome == "success":
            obs[0] = 4.9
            obs[4] = 5.0
            obs[6] = 0.3
            obs[7] = 1.0
        elif outcome == "collided":
            obs[0] = 1.0
            obs[4] = 5.0
            obs[6] = 0.3
            obs[7] = -0.1 if i == n - 1 else 0.5
        else:
            obs[0] = 0.5
            obs[4] = 5.0
            obs[6] = 0.3
            obs[7] = 0.8
        steps.append(
            StepRecord(
                t=i,
                action=np.zeros(2),
                obs=obs,
                reward_raw=0.0,
                done=i == n - 1,
                success=outcome == "success" and i == n - 1,
                collided=outcome == "collided" and i == n - 1,
                timeout=outcome == "timeout" and i == n - 1,
                robot=robot,
                humans=[],
                min_separation=float(obs[7]),
                potential=0.0,
            )
        )
    return Episode(
        schema_version=1,
        scenario=ScenarioSpec(
            scenario_id=sid,
            task_name="circle-crossing-v0",
            this_seed=0,
            phase="test",
            test_case=abs(hash(sid)) % 1000,
            human_num=5,
            time_step=0.25,
            time_limit=25.0,
            randomization="none",
            difficulty={"humans_near_path": float(near)},
        ),
        policy_name="toy",
        steps=steps,
        labels=EpisodeLabels(outcome=outcome),
    )


def test_extract_inject_roundtrip():
    vals, sites = extract_coeffs(REWARD)
    assert len(vals) >= 3
    assert sites
    scaled = [v * 1.5 for v in vals]
    new_code = inject_coeffs(REWARD, scaled)
    vals2, _ = extract_coeffs(new_code)
    assert len(vals2) == len(vals)
    for a, b in zip(vals2, scaled):
        assert abs(a - b) < 1e-6


def test_optimize_improves_or_holds_on_toy():
    train = [
        _ep("t0", "success", 0.0),
        _ep("t1", "success", 0.0),
        _ep("t2", "timeout", 1.0),
        _ep("t3", "collided", 2.0),
        _ep("t4", "timeout", 1.0),
        _ep("t5", "collided", 3.0),
    ]
    val = [
        _ep("v0", "success", 0.0),
        _ep("v1", "timeout", 1.0),
        _ep("v2", "collided", 2.0),
    ]
    result = optimize_reward_coefficients(
        REWARD,
        train,
        val_episodes=val,
        method="random",
        max_evals=12,
        seed=0,
    )
    assert result.n_evals >= 1
    assert result.tuned_code
    assert result.train_composite_after >= result.train_composite_before - 1e-6


def test_running_scale_normalizer():
    n = RunningScaleNormalizer()
    a = n.normalize_and_update({"progress": 1.0, "stall": -1.0})
    b = n.normalize_and_update({"progress": 3.0, "stall": -3.0})
    assert "progress" in a and "progress" in b
    assert n.counts["progress"] == 2


def test_potential_based_progress_positive_when_closer():
    pre = np.zeros(16)
    pre[0], pre[4] = 0.0, 5.0
    cur = np.zeros(16)
    cur[0], cur[4] = 2.0, 5.0
    assert potential_based_progress_value(pre, cur) > 0.0


def test_structure_prompt_nonempty():
    assert "STRUCTURE" in STRUCTURE_ONLY_PROMPT or "structure" in STRUCTURE_ONLY_PROMPT.lower()
