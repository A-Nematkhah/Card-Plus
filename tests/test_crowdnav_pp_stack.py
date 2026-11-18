"""CrowdNav++ / EvoNav parity stack: config, GST paths, dual-view reward."""
from __future__ import annotations

import numpy as np
import pytest

from code_generation.self_reflection.crowdnav_env_utils import build_config
from code_generation.self_reflection.crowdnav_gst import (
    GST_RAND_NAME,
    gst_checkpoint_ready,
    gst_model_dir,
)


def test_build_config_crowdnav_pp_flags(monkeypatch):
    monkeypatch.delenv("CARD_GST", raising=False)
    cfg = build_config("circle-crossing-v0", stack="crowdnav_pp")
    # Default: neural GST off, const-vel preds, selfAttn SRNN on.
    assert cfg.sim.predict_method == "const_vel"
    assert cfg.env.use_wrapper is False
    assert cfg.robot.policy == "selfAttn_merge_srnn"
    assert cfg.robot.visible is False
    assert cfg.sim.human_num == 20
    assert cfg.card_stack == "crowdnav_pp"
    assert cfg.pred.model_dir == ""

    monkeypatch.setenv("CARD_GST", "1")
    cfg_gst = build_config("circle-crossing-v0", stack="crowdnav_pp")
    assert cfg_gst.sim.predict_method == "inferred"
    assert cfg_gst.env.use_wrapper is True
    assert cfg_gst.robot.policy == "selfAttn_merge_srnn"
    assert GST_RAND_NAME in cfg_gst.pred.model_dir.replace("\\", "/")
    assert cfg_gst.pred.model_dir == gst_model_dir(randomized=True)


def test_build_config_flat_stack_keeps_no_gst():
    cfg = build_config("circle-crossing-v0", stack="flat")
    assert cfg.sim.predict_method == "none"
    assert cfg.env.use_wrapper is False
    assert cfg.robot.policy == "srnn"
    assert cfg.pred.model_dir == ""
    assert cfg.card_stack == "flat"


def test_dual_view_reward_overrides_scalar():
    import gym
    from code_generation.self_reflection.crowdnav_pp_env import DualViewRewardWrapper
    from code_generation.self_reflection.crowdnav_env_utils import OBS_DIM

    class _FakeSim(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Dict(
                {
                    "robot_node": gym.spaces.Box(-1, 1, shape=(1, 7), dtype=np.float32),
                    "temporal_edges": gym.spaces.Box(-1, 1, shape=(1, 2), dtype=np.float32),
                    "spatial_edges": gym.spaces.Box(-1, 1, shape=(20, 12), dtype=np.float32),
                    "visible_masks": gym.spaces.Box(0, 1, shape=(20,), dtype=np.bool_),
                    "detected_human_num": gym.spaces.Box(0, 20, shape=(1,), dtype=np.float32),
                }
            )
            self.action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
            self.time_limit = 50
            self.time_step = 0.25
            self.robot = type(
                "R",
                (),
                {
                    "px": 0.0,
                    "py": 0.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "gx": 1.0,
                    "gy": 0.0,
                    "radius": 0.3,
                    "FOV": 2.0,
                    "sensor_range": 5.0,
                },
            )()
            self.humans = []

        def reset(self):
            return {k: self.observation_space[k].sample() for k in self.observation_space.spaces}

        def step(self, action):
            del action
            ob = self.reset()
            info = {"info": type("ReachGoal", (), {"__str__": lambda self: "Reaching goal"})()}
            return ob, 99.0, True, info

        def talk2Env(self, data):
            return True

    env = DualViewRewardWrapper(_FakeSim(), use_ground_truth_reward=False, max_episode_steps=10)

    def _reward(self, action, obs):
        del action
        assert obs.shape[0] == OBS_DIM
        return -3.5, {"dummy": -3.5}

    DualViewRewardWrapper.compute_dense_reward = _reward
    env.reset()
    ob, rew, done, info = env.step(np.zeros(2, dtype=np.float32))
    assert isinstance(ob, dict)
    assert "robot_node" in ob
    assert float(rew) == -3.5
    assert float(info["llm_reward"]) == -3.5
    assert float(info["ground_truth_reward"]) == 99.0
    assert info["flat_obs"].shape[0] == OBS_DIM
    assert done is True
    assert "goal_progress" in info
    assert "proxy_score" in info
    assert float(info["proxy_score"]) == 1.0  # ReachGoal → success maps to 1


def test_smoke_reset_const_vel_no_gst(monkeypatch):
    monkeypatch.delenv("CARD_GST", raising=False)
    from code_generation.self_reflection.crowdnav_pp_env import smoke_reset

    out = smoke_reset()
    assert out["ok"] is True
    assert "robot_node" in out["obs_keys"]
    assert out["predict_method"] == "const_vel"
    assert out["robot_policy"] == "selfAttn_merge_srnn"
    assert out["robot_visible"] is False
    assert out["human_num"] == 20


@pytest.mark.skipif(not gst_checkpoint_ready(randomized=True), reason="GST checkpoint not downloaded")
def test_smoke_reset_with_gst(monkeypatch):
    monkeypatch.setenv("CARD_GST", "1")
    from code_generation.self_reflection.crowdnav_pp_env import smoke_reset

    out = smoke_reset()
    assert out["ok"] is True
    assert "robot_node" in out["obs_keys"]
    assert out["predict_method"] == "inferred"
    assert out["robot_visible"] is False
    assert out["human_num"] == 20
