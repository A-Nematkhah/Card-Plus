"""CrowdNav++ dual-view env for CARD: Dict+GST for policy, flat features for LLM rewards.

Avoids OpenAI ``baselines`` (not required) and Windows-incompatible ``fork`` ShmemVecEnv.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional, Tuple

import gym
import numpy as np
import torch

from code_generation.self_reflection.crowdnav_env_utils import (
    CROWDNAV_ROOT,
    SUB_REWARD_PREFIX_LOCAL,
    _ensure_crowdnav_path,
    _episode_flags,
    build_config,
    extract_feedback_state_from_obs,
    flatten_observation,
    load_reward_function,
)
from code_generation.self_reflection.crowdnav_gst import ensure_gst_checkpoint
from code_generation.self_reflection.utils import INFO_PREFIX


class DualViewRewardWrapper(gym.Wrapper):
    """Policy gets Dict obs; LLM reward uses flat ``flatten_observation`` features."""

    def __init__(
        self,
        env: gym.Env,
        *,
        use_ground_truth_reward: bool = True,
        max_episode_steps: Optional[int] = None,
    ) -> None:
        super().__init__(env)
        self.use_ground_truth_reward = bool(use_ground_truth_reward)
        self.pre_obs: Optional[np.ndarray] = None
        self._elapsed_steps = 0
        self._episode_d0 = 0.0
        self._episode_d_best = 0.0
        sim_horizon = int(getattr(env, "time_limit", 50) / getattr(env, "time_step", 0.25))
        self._max_episode_steps = int(max_episode_steps) if max_episode_steps else sim_horizon

    def compute_dense_reward(self, action, obs):
        assert False, "load_reward_function() must patch compute_dense_reward"

    def talk2Env(self, data):
        return self.env.talk2Env(data)

    def _flat_from_sim(self) -> np.ndarray:
        return flatten_observation(self.env)

    @staticmethod
    def _goal_dist(flat: np.ndarray) -> float:
        return float(np.hypot(float(flat[0] - flat[4]), float(flat[1] - flat[5])))

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        ob = self.env.reset(**kwargs) if kwargs else self.env.reset()
        self.pre_obs = self._flat_from_sim()
        self._episode_d0 = self._goal_dist(self.pre_obs)
        self._episode_d_best = float(self._episode_d0)
        return ob

    def step(self, action):
        from card_plus.progress import (
            crowdnav_goal_progress_from_dists,
            crowdnav_soft_near_goal,
            episode_proxy_score,
        )

        ob, rew, done, info = self.env.step(action)
        if not isinstance(info, dict):
            info = {"info": info}
        episode_info = info.get("info", info)
        success, collided, timeout = _episode_flags(episode_info)
        flat = self._flat_from_sim()
        d_now = self._goal_dist(flat)
        self._episode_d_best = min(float(self._episode_d_best), d_now)
        progress = crowdnav_goal_progress_from_dists(
            self._episode_d0, d_now, self._episode_d_best
        )
        soft_near = crowdnav_soft_near_goal(self._episode_d_best, radius=float(flat[6]))
        info["ground_truth_reward"] = float(rew)
        info["flat_obs"] = flat
        info["success"] = float(success)
        info["collided"] = float(collided)
        info["timeout"] = float(timeout)
        info["is_success"] = bool(success)
        info["goal_progress"] = float(progress)
        info["soft_near"] = float(soft_near)
        info["proxy_score"] = episode_proxy_score(
            success=float(success),
            progress=float(progress),
            collided=float(collided),
            soft_near=float(soft_near),
        )
        for key, value in extract_feedback_state_from_obs(flat).items():
            info[INFO_PREFIX + key] = value
            info[key] = value

        if self.pre_obs is None:
            self.pre_obs = flat
        if not self.use_ground_truth_reward:
            # Temporarily expose pre_obs for patched reward fns that read self.pre_obs.
            result = self.compute_dense_reward(action, flat)
            if isinstance(result, tuple):
                rew, rew_dict = result
                for key, value in rew_dict.items():
                    info[SUB_REWARD_PREFIX_LOCAL + key] = value
            else:
                rew = result
        info["llm_reward"] = float(rew)

        self._elapsed_steps += 1
        self.pre_obs = flat
        if success or collided:
            done = True
            info["TimeLimit.truncated"] = False
        elif self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
            info["timeout"] = 1.0
            info["bad_transition"] = True
        return ob, float(rew), bool(done), info


class VecPyTorch(torch.nn.Module):
    """Minimal torch conversion wrapper (CrowdNav VecPyTorch without baselines)."""

    # Not an nn.Module semantically; inherit only to avoid accidental name clash.
    pass


# Re-implement as VecEnvWrapper subclass without importing baselines.
def _make_vec_pytorch_cls():
    _ensure_crowdnav_path()
    from rl.vec_env.vec_env import VecEnvWrapper  # type: ignore

    class _VecPyTorch(VecEnvWrapper):
        def __init__(self, venv, device):
            super().__init__(venv)
            self.device = device

        def reset(self):
            obs = self.venv.reset()
            if isinstance(obs, dict):
                for key in obs:
                    obs[key] = torch.from_numpy(obs[key]).to(self.device)
            else:
                obs = torch.from_numpy(obs).float().to(self.device)
            return obs

        def step_async(self, actions):
            if isinstance(actions, torch.LongTensor):
                actions = actions.squeeze(1)
            if torch.is_tensor(actions):
                actions = actions.cpu().numpy()
            self.venv.step_async(actions)

        def talk2Env_wait(self):
            return self.venv.talk2Env_wait()

        def step_wait(self):
            obs, reward, done, info = self.venv.step_wait()
            if isinstance(obs, dict):
                for key in obs:
                    obs[key] = torch.from_numpy(obs[key]).to(self.device)
            else:
                obs = torch.from_numpy(obs).float().to(self.device)
            reward = torch.from_numpy(np.asarray(reward)).unsqueeze(dim=1).float()
            return obs, reward, done, info

    return _VecPyTorch


class VecRestoreLLMReward:
    """After GST reward shaping, restore LLM / declared reward from info."""

    def __init__(self, venv, *, enabled: bool = True):
        _ensure_crowdnav_path()
        from rl.vec_env.vec_env import VecEnvWrapper  # type: ignore

        class _Inner(VecEnvWrapper):
            def __init__(self, v, enabled_flag: bool):
                super().__init__(v)
                self.enabled_flag = enabled_flag

            def reset(self):
                return self.venv.reset()

            def talk2Env_wait(self):
                return self.venv.talk2Env_wait()

            def step_wait(self):
                obs, rews, dones, infos = self.venv.step_wait()
                if self.enabled_flag:
                    restored = []
                    for i, info in enumerate(infos):
                        if isinstance(info, dict) and "llm_reward" in info:
                            restored.append(float(info["llm_reward"]))
                        else:
                            # keep GST-shaped reward
                            if torch.is_tensor(rews):
                                restored.append(float(rews[i].item()))
                            else:
                                restored.append(float(np.asarray(rews).reshape(-1)[i]))
                    if torch.is_tensor(rews):
                        rews = torch.tensor(restored, dtype=rews.dtype, device=rews.device).unsqueeze(1)
                    else:
                        rews = np.asarray(restored, dtype=np.float32)
                return obs, rews, dones, infos

        self._inner = _Inner(venv, enabled)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _pp_env_id(config: SimpleNamespace) -> str:
    """Match CrowdNav++ env id to predict_method."""
    method = str(getattr(config.sim, "predict_method", "none"))
    if method == "inferred":
        return "CrowdSimPredRealGST-v0"
    if method in ("const_vel", "truth"):
        return "CrowdSimPred-v0"
    return "CrowdSimVarNum-v0"


def make_gst_base_env(
    config: SimpleNamespace,
    *,
    seed: int = 0,
    phase: str = "train",
    nenv: int = 1,
) -> gym.Env:
    _ensure_crowdnav_path()
    import crowd_sim  # noqa: F401  # registers env ids
    env = gym.make(_pp_env_id(config))
    # gym>=0.21 wraps with OrderEnforcing; CrowdNav attrs must land on the inner sim.
    core = env.unwrapped if hasattr(env, "unwrapped") else env
    core.configure(config)
    for target in (core, env):
        target.thisSeed = int(seed)
        target.nenv = int(nenv)
        target.phase = phase
    if hasattr(core, "seed"):
        core.seed(int(seed))
    elif hasattr(env, "seed"):
        env.seed(int(seed))
    return env


def make_dual_view_env(
    task_name: str = "circle-crossing-v0",
    *,
    seed: int = 0,
    phase: str = "train",
    use_ground_truth_reward: bool = True,
    reward_code: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    env_config: Optional[Dict[str, Any]] = None,
    ensure_gst: bool = True,
) -> DualViewRewardWrapper:
    config = build_config(task_name, env_config, stack="crowdnav_pp")
    use_gst = str(config.sim.predict_method) == "inferred"
    if ensure_gst and use_gst:
        with_random = bool(config.env.randomize_attributes)
        ensure_gst_checkpoint(randomized=with_random, download=True)
    # Align with make_pp_vec_env device policy (CARD_DEVICE / auto=CPU).
    pref = str(os.environ.get("CARD_DEVICE", "auto")).strip().lower()
    use_cuda = pref == "cuda" and torch.cuda.is_available()
    config.training.device = "cuda:0" if use_cuda else "cpu"
    config.args.cuda = bool(use_cuda)

    if (not use_ground_truth_reward) and reward_code:
        load_reward_function(code=reward_code)

    base = make_gst_base_env(config, seed=seed, phase=phase, nenv=1)
    return DualViewRewardWrapper(
        base,
        use_ground_truth_reward=use_ground_truth_reward,
        max_episode_steps=max_episode_steps,
    )


def make_pp_vec_env(
    task_name: str = "circle-crossing-v0",
    *,
    seed: int = 0,
    use_ground_truth_reward: bool = True,
    reward_code: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    env_config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    restore_llm_reward: bool = True,
):
    """Single-process DummyVecEnv + VecPyTorch + optional GST + LLM reward restore."""
    from code_generation.self_reflection.torch_compat import install_torch_load_compat

    # torch 1.11 rejects weights_only=; GST wrappers pass it for newer PyTorch.
    install_torch_load_compat()
    _ensure_crowdnav_path()
    # GST / crowd_sim imports resolve relative to CrowdNav root for some assets.
    os.chdir(CROWDNAV_ROOT)

    config = build_config(task_name, env_config, stack="crowdnav_pp")
    use_gst = str(config.sim.predict_method) == "inferred" and bool(config.env.use_wrapper)
    if use_gst:
        with_random = bool(config.env.randomize_attributes)
        ensure_gst_checkpoint(randomized=with_random, download=True)
    if device is None:
        # CARD_DEVICE=cpu|cuda|auto  (bench on RTX 3050 Laptop: CPU+4 threads was fastest)
        pref = str(os.environ.get("CARD_DEVICE", "auto")).strip().lower()
        if pref == "cpu":
            device = torch.device("cpu")
        elif pref == "cuda":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            # auto: prefer CPU for this stack (ORCA-bound); override with CARD_DEVICE=cuda
            device = torch.device("cpu")
    config.training.device = str(device) if device.type == "cpu" else "cuda:0"
    config.args.cuda = device.type == "cuda"
    config.env.num_processes = 1
    config.args.num_processes = 1

    def _thunk():
        return make_dual_view_env(
            task_name,
            seed=seed,
            phase="train",
            use_ground_truth_reward=use_ground_truth_reward,
            reward_code=reward_code,
            max_episode_steps=max_episode_steps,
            env_config=env_config,
            ensure_gst=False,
        )

    from rl.vec_env.dummy_vec_env import DummyVecEnv  # type: ignore

    VecPyTorchCls = _make_vec_pytorch_cls()
    envs = DummyVecEnv([_thunk])
    # DummyVecEnv.talk2Env expects env.env; DualView is a Wrapper so .env is the sim.
    envs = VecPyTorchCls(envs, device)
    if use_gst:
        from rl.vec_env.vec_pretext_normalize import VecPretextNormalize  # type: ignore

        envs = VecPretextNormalize(
            envs, gamma=float(config.reward.gamma), ob=False, ret=False, config=config, test=True
        )
    if restore_llm_reward and (not use_ground_truth_reward):
        envs = VecRestoreLLMReward(envs, enabled=True)._inner
    return envs, config, device


def smoke_reset(task_name: str = "circle-crossing-v0") -> Dict[str, Any]:
    """Reset once through the GST stack; used for verification."""
    envs, config, device = make_pp_vec_env(task_name, seed=0, use_ground_truth_reward=True)
    try:
        obs = envs.reset()
        keys = list(obs.keys()) if isinstance(obs, dict) else []
        return {
            "ok": True,
            "device": str(device),
            "obs_keys": keys,
            "predict_method": config.sim.predict_method,
            "robot_policy": config.robot.policy,
            "robot_visible": config.robot.visible,
            "human_num": config.sim.human_num,
        }
    finally:
        envs.close()
