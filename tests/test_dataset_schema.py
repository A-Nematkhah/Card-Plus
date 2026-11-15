"""Schema save/load round-trip (no simulator)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from card_plus.dataset.schema import (
    SCHEMA_VERSION,
    AgentState,
    Episode,
    EpisodeLabels,
    ScenarioSpec,
    StepRecord,
    load_episode,
    save_episode,
)


def _toy_episode() -> Episode:
    robot = AgentState(0, 1.0, 2.0, 0.0, 0.0, 0.3, -1.0, -2.0, True)
    human = AgentState(0, 0.5, 0.5, 0.1, 0.0, 0.3, 1.0, 1.0, False)
    scenario = ScenarioSpec(
        scenario_id="toy",
        task_name="circle-crossing-v0",
        this_seed=42,
        phase="test",
        test_case=0,
        human_num=1,
        time_step=0.25,
        time_limit=50.0,
        randomization="no_random",
        difficulty={"humans_near_path": 1.0},
        initial_robot=robot,
        initial_humans=[human],
    )
    step = StepRecord(
        t=0,
        action=np.array([0.1, -0.2]),
        obs=np.zeros(16 + 5 * 20),
        reward_raw=0.5,
        done=False,
        success=False,
        collided=False,
        timeout=False,
        robot=robot,
        humans=[human],
        min_separation=1.0,
        potential=-3.0,
    )
    return Episode(
        schema_version=SCHEMA_VERSION,
        scenario=scenario,
        policy_name="toy",
        steps=[step],
        labels=EpisodeLabels(outcome="incomplete"),
        initial_obs=np.zeros(16 + 5 * 20),
    )


def test_save_load_pickle_roundtrip(tmp_path: Path):
    ep = _toy_episode()
    path = save_episode(ep, tmp_path / "ep.pkl")
    loaded = load_episode(path)
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.scenario.scenario_id == "toy"
    assert loaded.T == 1
    assert np.allclose(loaded.steps[0].action, ep.steps[0].action)
    assert loaded.labels is not None and loaded.labels.outcome == "incomplete"


def test_save_load_json_roundtrip(tmp_path: Path):
    ep = _toy_episode()
    path = save_episode(ep, tmp_path / "ep.json")
    loaded = load_episode(path)
    assert loaded.scenario.test_case == 0
    assert np.allclose(loaded.obs_array(), ep.obs_array())
