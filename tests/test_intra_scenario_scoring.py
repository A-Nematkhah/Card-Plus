"""Intra-scenario ranking should discriminate rewards that only pass pair floors."""

import numpy as np

from card_plus.dataset.schema import AgentState, Episode, EpisodeLabels, ScenarioSpec, StepRecord
from card_plus.dataset.scoring import (
    calibrated_composite,
    intra_scenario_score,
    rule_quality_score,
)


def _steps(T: int) -> list:
    robot = AgentState(0, 0.0, 0.0, 0.0, 0.0, 0.3, 1.0, 0.0, True)
    out = []
    for i in range(max(T, 1)):
        out.append(
            StepRecord(
                t=i,
                action=np.zeros(2),
                obs=np.zeros(16),
                reward_raw=0.0,
                done=(i == T - 1),
                success=False,
                collided=False,
                timeout=False,
                robot=robot,
                humans=[],
                min_separation=1.0,
                potential=0.0,
            )
        )
    return out


def _ep(sid: str, outcome: str, T: int, progress: float, policy: str) -> Episode:
    return Episode(
        schema_version=1,
        scenario=ScenarioSpec(
            scenario_id=sid,
            task_name="circle-crossing-v0",
            this_seed=42,
            phase="test",
            test_case=0,
            human_num=10,
            time_step=0.25,
            time_limit=50.0,
            randomization="with_random",
            difficulty={},
        ),
        policy_name=policy,
        steps=_steps(T),
        labels=EpisodeLabels(outcome=outcome, progress=progress),
    )


def test_rule_prefers_success_over_collision():
    s = _ep("a", "success", 20, 1.0, "g")
    c = _ep("a", "collided", 10, 0.2, "r")
    assert rule_quality_score(s) > rule_quality_score(c)


def test_intra_perfect_when_returns_match_rule_order():
    eps = [
        _ep("s1", "success", 15, 1.0, "g"),
        _ep("s1", "timeout", 40, 0.4, "st"),
        _ep("s1", "collided", 8, 0.1, "r"),
    ]
    returns = [3.0, 1.0, -2.0]
    m = intra_scenario_score(eps, returns)
    assert m["intra_n_scenarios"] == 1.0
    assert m["intra_scenario"] > 0.9


def test_intra_low_when_returns_invert_rule_order():
    eps = [
        _ep("s1", "success", 15, 1.0, "g"),
        _ep("s1", "timeout", 40, 0.4, "st"),
        _ep("s1", "collided", 8, 0.1, "r"),
    ]
    returns = [-2.0, 1.0, 3.0]
    m = intra_scenario_score(eps, returns)
    assert m["intra_scenario"] < 0.3


def test_composite_weights_spread_more_than_rank_alone():
    low = calibrated_composite(
        offline_rank=0.99, separation=0.55, anti_hack=0.55, intra_scenario=0.45
    )
    high = calibrated_composite(
        offline_rank=0.99, separation=0.85, anti_hack=0.85, intra_scenario=0.90
    )
    assert high - low > 0.15
