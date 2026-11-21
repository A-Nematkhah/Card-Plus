"""Tests for offline sub-reward attribution + counterfactual helpers."""

from __future__ import annotations

import numpy as np

from card_plus.dataset.attribution import (
    failed_pair_attributions,
    format_attribution_block,
    pair_attribution,
)
from card_plus.dataset.counterfactual import run_counterfactual
from card_plus.dataset.rescore import load_reward_fn_from_code
from card_plus.dataset.schema import AgentState, Episode, EpisodeLabels, ScenarioSpec, StepRecord

REWARD_CODE = """
def compute_dense_reward(self, action, obs):
    import numpy as np
    progress = -0.1 * float(np.linalg.norm(obs[0:2] - obs[4:6]))
    collision = -50.0 if obs[7] < 0.0 else 0.0
    stall = -0.01
    total = progress + collision + stall
    return total, {"progress": progress, "collision": collision, "stall": stall}
"""


def _ep(outcome: str, n: int = 4, collide: bool = False) -> Episode:
    robot = AgentState(0, 0.0, 0.0, 0.0, 0.0, 0.3, 5.0, 0.0, True)
    steps = []
    for i in range(n):
        obs = np.zeros(116, dtype=np.float64)
        obs[0] = float(i) * 0.1
        obs[4] = 5.0
        obs[6] = 0.3
        obs[7] = -0.2 if collide and i == n - 1 else 0.8
        steps.append(
            StepRecord(
                t=i,
                action=np.asarray([0.1, 0.0], dtype=np.float64),
                obs=obs,
                reward_raw=0.0,
                done=i == n - 1,
                success=outcome == "success" and i == n - 1,
                collided=collide and i == n - 1,
                timeout=outcome == "timeout" and i == n - 1,
                robot=robot,
                humans=[],
                min_separation=float(obs[7]),
                potential=0.0,
            )
        )
    sc = ScenarioSpec(
        scenario_id=f"s_{outcome}",
        task_name="circle_crossing",
        this_seed=0,
        phase="test",
        test_case=0,
        human_num=5,
        time_step=0.25,
        time_limit=25.0,
        randomization="none",
    )
    return Episode(
        schema_version=1,
        scenario=sc,
        policy_name="p",
        steps=steps,
        labels=EpisodeLabels(outcome=outcome),
    )


def test_pair_attribution_has_components():
    eps = [
        _ep("timeout"),
        _ep("timeout"),
        _ep("collided", collide=True),
        _ep("collided", collide=True),
    ]
    fn = load_reward_fn_from_code(REWARD_CODE)
    attr = pair_attribution(eps, fn, "timeout/collided")
    assert attr["pair"] == "timeout/collided"
    names = {r["component"] for r in attr["components"]}
    assert "collision" in names or "progress" in names
    text = format_attribution_block([attr])
    assert "attribution" in text
    assert "timeout/collided" in text


def test_failed_pair_attributions_only_below_floor():
    eps = [_ep("timeout"), _ep("collided", collide=True)]
    fn = load_reward_fn_from_code(REWARD_CODE)
    attrs = failed_pair_attributions(
        eps,
        fn,
        pair_metrics={"timeout/collided": 0.1, "success/timeout": 0.99},
        floors={"timeout/collided": 0.6, "success/timeout": 0.8},
    )
    assert len(attrs) == 1
    assert attrs[0]["pair"] == "timeout/collided"


def test_counterfactual_runs():
    eps = [
        _ep("success"),
        _ep("timeout"),
        _ep("collided", collide=True),
    ]
    fn = load_reward_fn_from_code(REWARD_CODE)
    report = run_counterfactual(eps, fn)
    assert "timeout/collided" in report.baseline_pairs
    assert report.rows
    assert all(r.mode in ("drop", "scale2") for r in report.rows)
