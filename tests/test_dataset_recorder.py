"""Recorder: determinism, GT recompute, save/load (needs CrowdNav)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from card_plus.dataset.labels import attach_labels
from card_plus.dataset.policies import go_to_goal_policy, stationary_policy
from card_plus.dataset.recorder import (
    episodes_states_equal,
    gt_rewards_from_episode,
    record_episode,
    replay_actions,
)
from card_plus.dataset.rescore import rescore_episodes
from card_plus.dataset.scenarios import capture_scenario
from card_plus.dataset.schema import load_episode, save_episode


@pytest.fixture(scope="module")
def scenario0():
    return capture_scenario(0, this_seed=42)


def test_same_actions_same_states(scenario0):
    ep1 = record_episode(scenario0, go_to_goal_policy, policy_name="goto", max_steps=40)
    ep2 = replay_actions(scenario0, ep1.actions_array())
    assert episodes_states_equal(ep1, ep2, atol=1e-5)
    assert ep1.T == ep2.T
    assert ep1.T > 0


def test_gt_recompute_matches_reward_raw(scenario0):
    ep = record_episode(scenario0, go_to_goal_policy, policy_name="goto", max_steps=60)
    recomputed = gt_rewards_from_episode(ep)
    raw = ep.rewards_raw_array()
    assert recomputed.shape == raw.shape
    assert np.allclose(recomputed, raw, atol=1e-5, rtol=1e-5), (
        float(np.max(np.abs(recomputed - raw))),
        list(zip(recomputed[:5], raw[:5])),
    )


def test_stationary_labels_and_save_load(scenario0, tmp_path: Path):
    ep = record_episode(scenario0, stationary_policy, policy_name="stationary", max_steps=30)
    attach_labels(ep)
    assert ep.labels is not None
    assert ep.labels.outcome in {"collided", "timeout", "truncated", "success", "incomplete"}
    path = save_episode(ep, tmp_path / "stationary.pkl")
    loaded = load_episode(path)
    assert loaded.policy_name == "stationary"
    assert loaded.scenario.test_case == 0
    assert np.allclose(loaded.rewards_raw_array(), ep.rewards_raw_array())
    assert loaded.labels is not None


def test_rescore_gt_identity(scenario0):
    """Scoring with raw stored rewards via a passthrough is not needed;
    use a simple progress-shaped reward and ensure report runs."""
    ep_a = record_episode(scenario0, go_to_goal_policy, policy_name="goto", max_steps=40)
    ep_b = record_episode(
        capture_scenario(1, this_seed=42),
        stationary_policy,
        policy_name="stationary",
        max_steps=40,
    )
    attach_labels(ep_a)
    attach_labels(ep_b)

    def progress_reward(action, obs):
        del action
        # Negative distance to goal (higher/less negative when closer).
        return -float(np.linalg.norm(obs[0:2] - obs[4:6]))

    report = rescore_episodes([ep_a, ep_b], progress_reward)
    assert report.n_episodes == 2
    assert report.mean_return == report.mean_return  # not raising
    assert isinstance(report.to_dict()["pair_accuracies"], list)
