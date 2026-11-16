"""CrowdNav++ circle-crossing env for CARD (gym 0.21 + Box obs for SB3).

Aligned with the EvoNav paper simulator (arXiv:2605.11859 §5.1 / CrowdNav++ [27]):
square workspace ~12×12 m (arena_size=6), 20 ORCA humans that react to the robot,
sensor range 5 m, vmax=1 m/s, optional human attribute/goal randomization.

Vendors ``CrowdNav_Prediction_AttnGraph/`` (Shuijing725) and uses
``CrowdSimVarNum`` with ORCA humans. Does not import the upstream Config
class (it calls ``get_args()`` at import); builds a compatible namespace instead.
"""
from __future__ import annotations

import math
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np

from code_generation.self_reflection.utils import INFO_PREFIX

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CROWDNAV_ROOT = os.path.join(_REPO_ROOT, "CrowdNav_Prediction_AttnGraph")
_PATH_READY = False

SENSOR_RANGE = 5.0
# EvoNav / CrowdNav++ evaluate H ∈ {5,10,15,20}; pad to max so obs width is fixed.
MAX_HUMANS_PAD = 20
# CrowdNav++ default: circle on a square that spans ~12 m (paper W=[0,12]²).
EVONAV_ARENA_SIZE = 6.0
EVONAV_CIRCLE_RADIUS = 6.0 * math.sqrt(2)

CROWDNAV_TASKS = {
    # Default = EvoNav "with randomization" setting.
    "circle-crossing-v0": {
        "instruction": (
            "Navigate a holonomic robot from a random start to a random goal through a circle-crossing crowd "
            "(CrowdNav++ / EvoNav setting: 10 pedestrians on a ~12×12 m workspace). "
            "Reach the goal without colliding with any pedestrian. Keep a comfortable clearance; do not freeze in place "
            "or take a long detour when a shorter safe path exists. "
            "Humans are ORCA-controlled and react to the robot. "
            "The task is finished successfully when the robot arrives within its own radius of the goal without any collision."
        ),
        "human_num": 10,
        "circle_radius": EVONAV_CIRCLE_RADIUS,
        "arena_size": EVONAV_ARENA_SIZE,
        "time_limit": 50.0,
        "time_step": 0.25,
        "success_reward": 10.0,
        "collision_penalty": -20.0,
        "discomfort_dist": 0.25,
        "discomfort_penalty_factor": 10.0,
        "robot_visible": False,
        "humans_policy": "orca",
        # with_random | no_random  (EvoNav Table 1 two columns)
        "randomization": "with_random",
    },
}

# LLM abstract names -> expressions over ``obs`` (after RewardFunctionConverter).
# Longer keys are applied first so aliases like ``self.goal_distance`` do not
# break ``self.robot.goal_distance``.
_GOAL_DIST = "float(np.linalg.norm(obs[0:2] - obs[4:6]))"
_SPEED = "float(np.linalg.norm(obs[2:4]))"
mapping_dicts = {
    "self.robot.position": "obs[0:2]",
    "self.robot.velocity": "obs[2:4]",
    "self.robot.goal_position": "obs[4:6]",
    "self.robot.radius": "obs[6]",
    "self.robot.goal_distance": _GOAL_DIST,
    "self.robot.speed": _SPEED,
    # Common LLM shortcuts (must be shorter than the self.robot.* forms above).
    "self.goal_distance": _GOAL_DIST,
    "self.goal_position": "obs[4:6]",
    "self.position": "obs[0:2]",
    "self.velocity": "obs[2:4]",
    "self.speed": _SPEED,
    "self.min_separation": "obs[7]",
    "self.nearest_human.distance": "obs[8]",
    "self.nearest_human.relative_position": "obs[9:11]",
    "self.nearest_human.relative_velocity": "obs[11:13]",
    "self.second_nearest_human.distance": "obs[13]",
    "self.third_nearest_human.distance": "obs[14]",
    "self.detected_human_num": "obs[15]",
}

# Flat obs layout (16 + 5*MAX_HUMANS_PAD):
# 0:2 robot pos, 2:4 vel, 4:6 goal, 6 radius, 7 min_sep,
# 8 nearest dist, 9:11 nearest rel pos, 11:13 nearest rel vel,
# 13 second dist, 14 third dist, 15 detected count,
# then per-human [rel_px, rel_py, vx, vy, radius] sorted by distance.
OBS_DIM = 16 + 5 * MAX_HUMANS_PAD


def _ensure_crowdnav_path() -> None:
    global _PATH_READY
    if _PATH_READY:
        return
    if not os.path.isdir(CROWDNAV_ROOT):
        raise FileNotFoundError(
            f"CrowdNav++ vendor missing at {CROWDNAV_ROOT}. "
            "Clone https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph into the repo root."
        )
    root = os.path.abspath(CROWDNAV_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    _PATH_READY = True


def _ns(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


def build_config(task_name: str, env_config: Optional[Dict[str, Any]] = None) -> SimpleNamespace:
    if task_name not in CROWDNAV_TASKS:
        raise KeyError(f"Unknown CrowdNav task: {task_name}")
    task = dict(CROWDNAV_TASKS[task_name])
    if env_config:
        task.update(env_config)

    human_num = int(task["human_num"])
    if human_num > MAX_HUMANS_PAD:
        raise ValueError(f"human_num={human_num} exceeds MAX_HUMANS_PAD={MAX_HUMANS_PAD}")

    # EvoNav §5.1: with randomization → goal switches + v∈[0.5,1.5], r∈[0.3,0.5];
    # without → fixed traits, no mid-episode goal switches.
    randomization = str(task.get("randomization", "with_random"))
    with_random = randomization != "no_random"

    return _ns(
        args=_ns(cuda=False, num_processes=1, sort_humans=True),
        training=_ns(device="cpu"),
        env=_ns(
            time_limit=float(task["time_limit"]),
            time_step=float(task["time_step"]),
            # sample_random_attributes: v_pref~U(0.5,1.5), radius~U(0.3,0.5)
            randomize_attributes=with_random,
            val_size=100,
            test_size=500,
            num_processes=1,
            record=False,
            load_act=False,
            use_wrapper=False,
        ),
        reward=_ns(
            success_reward=float(task["success_reward"]),
            collision_penalty=float(task["collision_penalty"]),
            discomfort_dist=float(task["discomfort_dist"]),
            discomfort_penalty_factor=float(task["discomfort_penalty_factor"]),
            gamma=0.99,
        ),
        sim=_ns(
            circle_radius=float(task["circle_radius"]),
            arena_size=float(task["arena_size"]),
            human_num=human_num,
            human_num_range=0,
            predict_steps=5,
            # CARD flat-obs path; GST prediction left off (policy is SB3 MLP, not CrowdNav SRNN).
            predict_method="none",
            render=False,
        ),
        humans=_ns(
            visible=True,
            policy=str(task["humans_policy"]),
            radius=0.3,
            v_pref=1.0,
            sensor="coordinates",
            FOV=2.0,
            random_goal_changing=with_random,
            goal_change_chance=0.5,
            end_goal_changing=True,
            end_goal_change_chance=1.0,
            random_radii=False,
            random_v_pref=False,
            random_unobservability=False,
            unobservable_chance=0.3,
            random_policy_changing=False,
        ),
        robot=_ns(
            visible=bool(task["robot_visible"]),
            policy="srnn",
            radius=0.3,
            v_pref=1.0,
            sensor="coordinates",
            FOV=2.0,
            sensor_range=SENSOR_RANGE,
        ),
        action_space=_ns(kinematics="holonomic"),
        orca=_ns(neighbor_dist=10, safety_space=0.15, time_horizon=5, time_horizon_obst=5),
        sf=_ns(A=2.0, B=1.0, KI=1.0),
        data=_ns(
            tot_steps=40000,
            render=False,
            collect_train_data=False,
            num_processes=1,
            data_save_dir="",
            pred_timestep=0.25,
        ),
        pred=_ns(model_dir=""),
        lidar=_ns(angular_res=5, range=10),
        sim2real=_ns(
            use_dummy_detect=True,
            record=False,
            load_act=False,
            ROSStepInterval=0.03,
            fixed_time_interval=0.1,
            use_fixed_time_interval=True,
        ),
        render_traj=False,
        save_slides=False,
        save_path=None,
    )


def describe_action_space() -> str:
    return (
        "The action space of the robot is a continuous `Box(-1, 1, (2,), float32)` for holonomic "
        "(vx, vy) in m/s, later scaled by the robot preferred speed (1 m/s)."
    )


def _episode_flags(episode_info: Any) -> Tuple[bool, bool, bool]:
    name = type(episode_info).__name__
    return name == "ReachGoal", name == "Collision", name == "Timeout"


def _human_features(sim) -> Tuple[np.ndarray, float, float, float, float, np.ndarray, np.ndarray]:
    """Return padded human block + nearest/second/third distances and nearest rel pos/vel."""
    robot = sim.robot
    rpx, rpy = float(robot.px), float(robot.py)
    rvx, rvy = float(robot.vx), float(robot.vy)
    rr = float(robot.radius)
    rows = []
    for human in sim.humans:
        dx = float(human.px) - rpx
        dy = float(human.py) - rpy
        dist = math.hypot(dx, dy) - float(human.radius) - rr
        rows.append((dist, dx, dy, float(human.vx) - rvx, float(human.vy) - rvy, float(human.radius)))
    rows.sort(key=lambda item: item[0])

    pad = np.zeros((MAX_HUMANS_PAD, 5), dtype=np.float32)
    for i, (dist, dx, dy, dvx, dvy, radius) in enumerate(rows[:MAX_HUMANS_PAD]):
        pad[i, 0] = dx
        pad[i, 1] = dy
        pad[i, 2] = dvx
        pad[i, 3] = dvy
        pad[i, 4] = radius

    def dist_at(idx: int) -> float:
        return float(rows[idx][0]) if idx < len(rows) else 15.0

    if rows:
        nearest_rel = np.array([rows[0][1], rows[0][2]], dtype=np.float32)
        nearest_rel_v = np.array([rows[0][3], rows[0][4]], dtype=np.float32)
        min_sep = float(rows[0][0])
    else:
        nearest_rel = np.zeros(2, dtype=np.float32)
        nearest_rel_v = np.zeros(2, dtype=np.float32)
        min_sep = 15.0

    return pad, min_sep, dist_at(0), dist_at(1), dist_at(2), nearest_rel, nearest_rel_v


def flatten_observation(sim) -> np.ndarray:
    robot = sim.robot
    pad, min_sep, d0, d1, d2, nearest_rel, nearest_rel_v = _human_features(sim)
    out = np.empty(OBS_DIM, dtype=np.float32)
    out[0] = float(robot.px)
    out[1] = float(robot.py)
    out[2] = float(robot.vx)
    out[3] = float(robot.vy)
    out[4] = float(robot.gx)
    out[5] = float(robot.gy)
    out[6] = float(robot.radius)
    out[7] = min_sep
    out[8] = d0
    out[9] = nearest_rel[0]
    out[10] = nearest_rel[1]
    out[11] = nearest_rel_v[0]
    out[12] = nearest_rel_v[1]
    out[13] = d1
    out[14] = d2
    out[15] = float(len(sim.humans))
    out[16:] = pad.reshape(-1)
    return out


def extract_feedback_state_from_obs(obs: np.ndarray) -> Dict[str, Any]:
    """Build reward-feedback dict from a flat obs (no second sim scan)."""
    goal_dist = float(math.hypot(float(obs[0] - obs[4]), float(obs[1] - obs[5])))
    return {
        "self.robot.position": obs[0:2],
        "self.robot.velocity": obs[2:4],
        "self.robot.goal_position": obs[4:6],
        "self.robot.radius": float(obs[6]),
        "self.robot.goal_distance": goal_dist,
        "self.robot.speed": float(math.hypot(float(obs[2]), float(obs[3]))),
        "self.robot.reached_goal": bool(goal_dist < float(obs[6])),
        "self.min_separation": float(obs[7]),
        "self.nearest_human.distance": float(obs[8]),
        "self.nearest_human.relative_position": obs[9:11],
        "self.nearest_human.relative_velocity": obs[11:13],
        "self.second_nearest_human.distance": float(obs[13]),
        "self.third_nearest_human.distance": float(obs[14]),
        "self.detected_human_num": float(obs[15]),
    }


def extract_feedback_state(sim) -> Dict[str, Any]:
    return extract_feedback_state_from_obs(flatten_observation(sim))


class CrowdNavEnv(gym.Env):
    """gym 0.21 wrapper: Box obs/action around CrowdNav++ ``CrowdSimVarNum``."""

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, task_name: str = "circle-crossing-v0", env_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        _ensure_crowdnav_path()
        from crowd_sim.envs.crowd_sim_var_num import CrowdSimVarNum  # type: ignore

        self.task_name = task_name
        self.config = build_config(task_name, env_config)
        self._sim = CrowdSimVarNum()
        self._sim.configure(self.config)
        self._sim.thisSeed = 0
        self._sim.nenv = 1
        self._sim.phase = "train"

        self.max_episode_steps = int(self.config.env.time_limit / self.config.env.time_step)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._elapsed_steps = 0
        self._fig = None
        self._ax = None

    @property
    def sim(self):
        return self._sim

    def seed(self, seed=None):
        if seed is not None:
            self._sim.thisSeed = int(seed)
            np.random.seed(int(seed))
        return [seed]

    def reset(self):
        self._elapsed_steps = 0
        self._sim.reset(phase=self._sim.phase or "train")
        return flatten_observation(self._sim)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        # Map [-1,1] -> preferred-speed scaled holonomic velocity for clip_action.
        scaled = action * float(self._sim.robot.v_pref)
        ob_dict, reward, done, info = self._sim.step(scaled)
        del ob_dict  # dict obs unused; we flatten from sim state
        self._elapsed_steps += 1

        episode_info = info.get("info", info) if isinstance(info, dict) else info
        success, collided, timeout = _episode_flags(episode_info)
        obs = flatten_observation(self._sim)

        out_info: Dict[str, Any] = {
            "success": float(success),
            "collided": float(collided),
            "timeout": float(timeout),
            "ground_truth_reward": float(reward),
            "episode_info": str(episode_info),
        }
        # One pass: derive feedback from the same obs used as the transition observation.
        for key, value in extract_feedback_state_from_obs(obs).items():
            out_info[INFO_PREFIX + key] = value
            out_info[key] = value

        if self._elapsed_steps >= self.max_episode_steps and not done:
            done = True
            out_info["TimeLimit.truncated"] = True
            out_info["timeout"] = 1.0

        return obs, float(reward), bool(done), out_info

    def render(self, mode="human"):
        import matplotlib.pyplot as plt
        from matplotlib import patches

        if self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(5, 5))
        ax = self._ax
        ax.clear()
        arena = float(self.config.sim.arena_size) + 1.0
        ax.set_xlim(-arena, arena)
        ax.set_ylim(-arena, arena)
        ax.set_aspect("equal")
        ax.set_title(self.task_name)

        robot = self._sim.robot
        ax.add_patch(plt.Circle(robot.get_position(), robot.radius, color="gold", zorder=3))
        ax.plot([robot.gx], [robot.gy], marker="*", color="red", markersize=12)
        for human in self._sim.humans:
            ax.add_patch(plt.Circle((human.px, human.py), human.radius, color="blue", alpha=0.6))
            ax.add_patch(
                patches.FancyArrowPatch(
                    (human.px, human.py),
                    (human.px + human.vx * 0.3, human.py + human.vy * 0.3),
                    color="navy",
                    arrowstyle="->",
                    mutation_scale=8,
                )
            )
        self._fig.canvas.draw()
        if mode == "rgb_array":
            # Matplotlib Agg: tostring_rgb removed; prefer RGBA buffer then drop alpha.
            canvas = self._fig.canvas
            width, height = canvas.get_width_height()
            if hasattr(canvas, "buffer_rgba"):
                buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
                return np.ascontiguousarray(buf[:, :, :3])
            if hasattr(canvas, "tostring_rgb"):
                buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
                return buf.reshape(height, width, 3)
            argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape(height, width, 4)
            return np.ascontiguousarray(argb[:, :, [1, 2, 3]])
        return None

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None


def make_crowdnav_env(task_name: str, env_config: Optional[Dict[str, Any]] = None) -> CrowdNavEnv:
    return CrowdNavEnv(task_name=task_name, env_config=env_config)


# --- SB3 / Subproc-safe wrappers (no torch import: workers stay light) ---

SUB_REWARD_PREFIX_LOCAL = "sub_reward/"


class ContinuousTaskWrapper(gym.Wrapper):
    def __init__(self, args, env, max_episode_steps: int, is_eval: bool, use_ground_truth_reward: bool) -> None:
        super().__init__(env)
        self._elapsed_steps = 0
        self.pre_obs = None
        self.args = args
        self._max_episode_steps = max_episode_steps
        self.is_eval = is_eval
        self.use_ground_truth_reward = use_ground_truth_reward

    def reset(self):
        self._elapsed_steps = 0
        self.pre_obs = super().reset()
        return self.pre_obs

    def compute_dense_reward(self, action, obs):
        assert False

    def step(self, action):
        ob, rew, done, info = super().step(action)
        if not self.use_ground_truth_reward:
            # ``self.pre_obs`` is the previous flat obs (for delta goal-distance).
            if getattr(self, "pre_obs", None) is None:
                self.pre_obs = ob
            result = self.compute_dense_reward(action, ob)
            if isinstance(result, tuple):
                rew, rew_dict = result
                for key, value in rew_dict.items():
                    info[SUB_REWARD_PREFIX_LOCAL + key] = value
            else:
                rew = result
        info["ground_truth_reward"] = rew
        self._elapsed_steps += 1
        self.pre_obs = ob
        # Keep simulator terminals (success / collision). Only force-end on timeout.
        reached = bool(info.get("success", 0)) or bool(info.get("is_success", 0))
        collided = bool(info.get("collided", 0))
        if reached or collided:
            done = True
            info["TimeLimit.truncated"] = False
        elif self._elapsed_steps >= self._max_episode_steps:
            done = True
            info["TimeLimit.truncated"] = True
            info["timeout"] = 1.0
        return ob, rew, bool(done), info


class SuccessInfoWrapper(gym.Wrapper):
    def step(self, action):
        ob, rew, done, info = super().step(action)
        info["is_success"] = bool(info.get("success", 0))
        if info["is_success"]:
            done = True
        return ob, rew, done, info


def load_reward_function(code: str) -> None:
    """Patch ContinuousTaskWrapper in *this* process (must run inside each Subproc worker)."""
    namespace = {**globals(), "np": np, "Dict": Dict, "Tuple": Tuple}
    exec(code, namespace)
    ContinuousTaskWrapper.compute_dense_reward = namespace["compute_dense_reward"]


def make_env(
    args,
    seed,
    max_episode_steps: int = None,
    record_dir: str = None,
    is_eval: bool = False,
    use_ground_truth_reward: bool = True,
    reward_code: Optional[str] = None,
):
    """Factory for Dummy/SubprocVecEnv. Loads reward_code inside the worker process."""
    task_name = args.task_name

    def _init() -> gym.Env:
        if (not use_ground_truth_reward) and reward_code:
            load_reward_function(code=reward_code)
        env = make_crowdnav_env(task_name)
        env.seed(seed)
        sim_horizon = int(getattr(env, "max_episode_steps", 200))
        horizon = int(max_episode_steps) if max_episode_steps is not None else sim_horizon
        if not is_eval and max_episode_steps is not None:
            env = ContinuousTaskWrapper(
                args, env, horizon, is_eval=is_eval, use_ground_truth_reward=use_ground_truth_reward
            )
        elif is_eval and record_dir is not None:
            env = ContinuousTaskWrapper(
                args, env, horizon, is_eval=is_eval, use_ground_truth_reward=use_ground_truth_reward
            )
            env = SuccessInfoWrapper(env)
        else:
            raise RuntimeError("Environmental parameter conflicts!")
        return env

    return _init
