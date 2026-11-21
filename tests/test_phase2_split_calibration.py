"""Phase-2 split + calibration + cautious policy unit tests."""

from __future__ import annotations

import json

import numpy as np

from card_plus.dataset.calibration import (
    ZooEntryScore,
    calibrate_reward_zoo,
    kendall_tau,
)
from card_plus.dataset.policies import cautious_orca_policy, get_policy
from card_plus.dataset.reward_zoo import mutate_reward_variants, save_zoo_json, load_zoo_json
from card_plus.dataset.schema import AgentState, Episode, EpisodeLabels, ScenarioSpec, StepRecord
from card_plus.dataset.split import (
    ensure_dataset_split,
    filter_episodes_by_split,
    stratified_train_val_ids,
)
from card_plus.tpe_v2 import spearman_rho


def _ep(sid: str, near: float, outcome: str = "timeout") -> Episode:
    robot = AgentState(0, 0.0, 0.0, 0.0, 0.0, 0.3, 5.0, 0.0, True)
    step = StepRecord(
        t=0,
        action=np.zeros(2),
        obs=np.zeros(116),
        reward_raw=0.0,
        done=True,
        success=outcome == "success",
        collided=outcome == "collided",
        timeout=outcome == "timeout",
        robot=robot,
        humans=[],
        min_separation=1.0,
        potential=0.0,
    )
    return Episode(
        schema_version=1,
        scenario=ScenarioSpec(
            scenario_id=sid,
            task_name="circle-crossing-v0",
            this_seed=42,
            phase="test",
            test_case=hash(sid) % 10000,
            human_num=15,
            time_step=0.25,
            time_limit=50.0,
            randomization="with_random",
            difficulty={"humans_near_path": float(near)},
        ),
        policy_name="toy",
        steps=[step],
        labels=EpisodeLabels(outcome=outcome),
    )


def test_stratified_split_nonempty_both_sides():
    eps = [_ep(f"s{i}", near=float(i % 5), outcome="timeout") for i in range(20)]
    # Duplicate policies same scenario should not duplicate ids
    eps += [_ep(f"s{i}", near=float(i % 5), outcome="success") for i in range(20)]
    split = stratified_train_val_ids(eps, val_fraction=0.25, seed=1)
    assert split["train"]
    assert split["val"]
    assert set(split["train"]).isdisjoint(set(split["val"]))
    assert len(split["train"]) + len(split["val"]) == 20


def test_ensure_split_persists(tmp_path):
    eps = [_ep(f"s{i}", near=float(i)) for i in range(12)]
    split = ensure_dataset_split(tmp_path, eps, seed=0)
    path = tmp_path / "scenario_split.json"
    assert path.is_file()
    again = ensure_dataset_split(tmp_path, eps, seed=99)  # should reuse file
    assert again["train"] == split["train"]
    train_eps = filter_episodes_by_split(eps, split, "train")
    assert train_eps
    assert all(e.scenario.scenario_id in set(split["train"]) for e in train_eps)


def test_cautious_policy_avoids_when_close():
    obs = np.zeros(116, dtype=np.float64)
    obs[0:2] = [0.0, 0.0]
    obs[4:6] = [5.0, 0.0]
    obs[7] = 0.2  # tight
    obs[9:11] = [1.0, 0.0]  # human ahead
    a = cautious_orca_policy(obs)
    # Should push away from human (negative x) more than pure goal chase
    goal = get_policy("go_to_goal")(obs)
    assert a[0] < goal[0]


def test_get_policy_cautious():
    fn = get_policy("cautious")
    a = fn(np.zeros(116))
    assert a.shape == (2,)


def test_kendall_and_calibration_go():
    # Perfect monotone relation → high Spearman/Kendall
    entries = [
        ZooEntryScore(
            reward_id=f"r{i}",
            composite=0.1 * i,
            metrics={
                "offline_rank": 0.1 * i,
                "separation": 0.1 * i,
                "anti_hack": 0.1 * i,
                "intra_scenario": 0.1 * i,
            },
            online_success=0.05 * i,
        )
        for i in range(1, 25)
    ]
    report = calibrate_reward_zoo(entries, spearman_threshold=0.5, min_n=20)
    assert report.n == 24
    assert report.spearman >= 0.5
    assert report.kendall >= 0.5
    assert report.go is True
    assert "full" in report.component_ablation


def test_calibration_nogo_uncorrelated():
    entries = [
        ZooEntryScore(
            reward_id=f"r{i}",
            composite=float(i % 3),
            metrics={
                "offline_rank": 0.5,
                "separation": 0.5,
                "anti_hack": 0.5,
                "intra_scenario": 0.5,
            },
            online_success=float((i * 7) % 5),
        )
        for i in range(25)
    ]
    # May or may not be low; just ensure API returns and respects min_n
    report = calibrate_reward_zoo(entries, spearman_threshold=0.99, min_n=20)
    assert report.n == 25
    assert report.go is False


def test_mutate_and_zoo_roundtrip(tmp_path):
    code = '''
def compute_dense_reward(self, action, obs):
    progress = -0.1 * float(obs[0])
    collision = -20.0
    total = progress + collision
    return total, {"progress": progress, "collision": collision}
'''
    variants = mutate_reward_variants(code, base_id="toy")
    assert len(variants) >= 2
    path = save_zoo_json(variants, tmp_path / "zoo.json")
    loaded = load_zoo_json(path)
    assert len(loaded) == len(variants)
    assert loaded[0].reward_id == "toy"


def test_spearman_matches_perfect():
    x = np.arange(10, dtype=np.float64)
    y = x * 2 + 1
    assert spearman_rho(x, y) > 0.99
    assert kendall_tau(x, y) > 0.99
