"""Section-1 CARD core: TPE v1 edge cases, traj end text, eval feedback."""

from __future__ import annotations

import numpy as np

from code_generation.post_process.post_process import RewardFunctionConverter
from code_generation.self_reflection.utils import (
    convert_traj_to_str,
    get_eval_feedback,
    trajectories_return_check,
)


def _traj(*, success: bool, collided: bool = False, timeout: bool = False, length: int = 8, reward: float = 1.0):
    rewards = np.full(length, reward, dtype=np.float64)
    info = {
        "success": float(success),
        "is_success": success,
        "collided": float(collided),
        "timeout": float(timeout),
    }
    return {
        "obs": np.zeros((length, 4)),
        "actions": np.zeros((length, 2)),
        "rewards": rewards,
        "dones": np.zeros(length, dtype=bool),
        "infos": [info] * length,
        "self.robot.position": np.zeros((length, 2)),
    }


def _const_reward_fn(**kwargs):
    return 1.0, {"dummy": 1.0}


def test_tpe_v1_all_fail_does_not_pass():
    trajs = [_traj(success=False, collided=True, reward=0.0) for _ in range(6)]
    correct_count, n_succ, rate, passed, feedback = trajectories_return_check(
        reward_function=_const_reward_fn,
        trajectories=trajs,
        mapping_dicts={},
        threshold=0.8,
        log_interval=10,
        benchmark_name="crowdnav",
    )
    assert n_succ == 0
    assert correct_count == 0
    assert rate == 0.0
    assert passed is False
    assert "all trajectories failed" in feedback


def test_tpe_v1_all_success_passes_vacuously():
    trajs = [_traj(success=True, reward=2.0) for _ in range(5)]
    correct_count, n_succ, rate, passed, feedback = trajectories_return_check(
        reward_function=_const_reward_fn,
        trajectories=trajs,
        mapping_dicts={},
        threshold=0.8,
        log_interval=10,
        benchmark_name="crowdnav",
    )
    assert n_succ == 5
    assert rate == 1.0
    assert passed is True
    assert "all trajectories succeeded" in feedback


def test_traj_end_mentions_collision_not_timeout():
    traj = _traj(success=False, collided=True, length=5)
    texts = convert_traj_to_str(
        [traj],
        mapping_dicts={"self.robot.position": "self.robot.position"},
        log_interval=10,
        benchmark_name="crowdnav",
    )
    assert "collided" in texts[0]
    assert "maximum trajectory length" not in texts[0]


def test_traj_end_mentions_timeout_when_flagged():
    traj = _traj(success=False, timeout=True, length=5)
    texts = convert_traj_to_str(
        [traj],
        mapping_dicts={},
        log_interval=10,
        benchmark_name="crowdnav",
    )
    assert "maximum trajectory length" in texts[0]


def test_get_eval_feedback_single_traj_not_duplicated():
    traj = _traj(success=True, length=4, reward=3.0)
    fb = get_eval_feedback(
        [traj],
        mapping_dicts={},
        log_interval=10,
        benchmark_name="crowdnav",
    )
    assert "Only one distinct evaluation trajectory" in fb
    assert fb.count("trajectory with return") == 1


def test_converter_does_not_replace_identifier_substrings():
    conv = RewardFunctionConverter({"pos": "robot_pos"})
    # "compose" contains "pos" as substring; word-boundary replace must leave it.
    src = "compose = pos + 1"
    out = conv.general_to_specific(src)
    assert "compose" in out
    assert "robot_pos" in out
    assert "comrobot_pos" not in out
