"""Section 4: CrowdNav env adapter + xAI key_manager fixes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from code_generation.self_reflection.crowdnav_env_utils import (
    CROWDNAV_TASKS,
    SENSOR_RANGE,
    ContinuousTaskWrapper,
    _detected_human_num,
    _episode_flags,
    _human_in_robot_fov,
    build_config,
    flatten_observation,
)
from key_manager import _is_non_retryable_client_error


class _FakeExc(Exception):
    def __init__(self, status_code=None, message=""):
        super().__init__(message)
        self.status_code = status_code


def test_robot_visible_matches_evonav_instruction():
    task = CROWDNAV_TASKS["circle-crossing-v0"]
    assert task["robot_visible"] is False
    assert task["human_num"] == 20
    assert "not to the robot" in task["instruction"]
    cfg = build_config("circle-crossing-v0")
    assert cfg.robot.visible is False
    assert cfg.sim.human_num == 20
    # Default stack remains flat; crowdnav_pp keeps selfAttn (GST off by default).
    assert cfg.sim.predict_method == "none"
    pp = build_config("circle-crossing-v0", stack="crowdnav_pp")
    assert pp.sim.predict_method == "const_vel"
    assert pp.env.use_wrapper is False
    assert pp.robot.policy == "selfAttn_merge_srnn"


def test_episode_flags_class_name_and_str_fallback():
    class ReachGoal:
        def __str__(self):
            return "Reaching goal"

    class Collision:
        def __str__(self):
            return "Collision"

    class Timeout:
        def __str__(self):
            return "Timeout"

    assert _episode_flags(ReachGoal()) == (True, False, False)
    assert _episode_flags(Collision()) == (False, True, False)
    assert _episode_flags(Timeout()) == (False, False, True)

    # Renamed upstream class but stable __str__
    class GoalReached:
        def __str__(self):
            return "Reaching goal"

    assert _episode_flags(GoalReached()) == (True, False, False)
    assert _episode_flags({"success": True, "collided": False, "timeout": False}) == (True, False, False)
    assert _episode_flags({"collision": True}) == (False, True, False)


def test_detected_human_num_uses_sensor_range_not_total_count():
    robot = SimpleNamespace(px=0.0, py=0.0, vx=1.0, vy=0.0, radius=0.3, FOV=2.0, sensor_range=SENSOR_RANGE)
    near = SimpleNamespace(px=1.0, py=0.0, vx=0.0, vy=0.0, radius=0.3)
    far = SimpleNamespace(px=20.0, py=0.0, vx=0.0, vy=0.0, radius=0.3)
    sim = SimpleNamespace(robot=robot, humans=[near, far], robot_fov=2.0 * np.pi)

    assert _human_in_robot_fov(robot, near, sensor_range=SENSOR_RANGE, fov_rad=2.0 * np.pi)
    assert not _human_in_robot_fov(robot, far, sensor_range=SENSOR_RANGE, fov_rad=2.0 * np.pi)
    assert _detected_human_num(sim) == 1.0

    # Prefer CrowdNav get_num_human_in_fov when present
    sim.get_num_human_in_fov = lambda: ([near], 1, [True, False])
    assert _detected_human_num(sim) == 1.0


def test_flatten_obs15_is_detected_count():
    robot = SimpleNamespace(
        px=0.0, py=0.0, vx=1.0, vy=0.0, gx=5.0, gy=0.0, radius=0.3, FOV=2.0, sensor_range=SENSOR_RANGE
    )
    near = SimpleNamespace(px=1.0, py=0.0, vx=0.0, vy=0.0, radius=0.3)
    far = SimpleNamespace(px=20.0, py=0.0, vx=0.0, vy=0.0, radius=0.3)
    sim = SimpleNamespace(robot=robot, humans=[near, far], robot_fov=2.0 * np.pi)
    obs = flatten_observation(sim)
    assert obs.shape[0] >= 16
    assert float(obs[15]) == 1.0
    assert len(sim.humans) == 2


def test_continuous_wrapper_preserves_ground_truth_reward():
    import gym

    class _Inner(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        def reset(self):
            return np.zeros(4, dtype=np.float32)

        def step(self, action):
            info = {"success": 0.0, "collided": 0.0, "ground_truth_reward": 7.5}
            return np.ones(4, dtype=np.float32), 7.5, False, info

    class _Wrapped(ContinuousTaskWrapper):
        def compute_dense_reward(self, action, obs):
            return -1.0, {"toy": 0.5}

    env = _Wrapped(SimpleNamespace(), _Inner(), max_episode_steps=10, is_eval=False, use_ground_truth_reward=False)
    env.reset()
    _ob, rew, _done, info = env.step(np.zeros(2))
    assert rew == -1.0
    assert info["ground_truth_reward"] == 7.5
    assert info["sub_reward/toy"] == 0.5


def test_xai_402_spending_limit_is_non_retryable():
    assert _is_non_retryable_client_error(_FakeExc(402, "Payment Required"))
    assert _is_non_retryable_client_error(_FakeExc(None, "spending-limit exceeded"))
    assert _is_non_retryable_client_error(_FakeExc(401, "unauthorized"))
    assert not _is_non_retryable_client_error(_FakeExc(429, "rate limit"))
    assert not _is_non_retryable_client_error(_FakeExc(503, "unavailable"))
