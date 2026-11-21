"""Policy callables: ``policy_fn(obs) -> action`` in Box([-1,1]^2)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

PathLike = Union[str, Path]
PolicyFn = Callable[[np.ndarray], np.ndarray]


def _clip_action(action: np.ndarray) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).reshape(2)
    return np.clip(a, -1.0, 1.0).astype(np.float32)


def go_to_goal_policy(obs: np.ndarray, *, speed: float = 1.0) -> np.ndarray:
    """Direction toward goal in obs frame (obs[0:2] pos, obs[4:6] goal)."""
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    delta = obs[4:6] - obs[0:2]
    n = float(np.linalg.norm(delta))
    if n < 1e-8:
        return np.zeros(2, dtype=np.float32)
    scale = float(np.clip(speed, 0.0, 1.0))
    return _clip_action((delta / n) * scale)


def cautious_orca_policy(
    obs: np.ndarray,
    *,
    speed: float = 0.7,
    comfort: float = 0.8,
    avoid_gain: float = 1.2,
) -> np.ndarray:
    """ORCA-like blend: go-to-goal, plus repulsion when min_sep is tight.

    Uses ``obs[7]`` (min separation) and nearest-human relative ``obs[9:11]``.
    Not a full ORCA solver — a lightweight cautious baseline for dataset v2.
    """
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    goal = go_to_goal_policy(obs, speed=speed)
    if obs.shape[0] < 11:
        return goal
    min_sep = float(obs[7])
    if min_sep >= float(comfort):
        return goal
    rel = obs[9:11]
    n = float(np.linalg.norm(rel))
    if n < 1e-8:
        return goal
    # Stronger avoidance as clearance shrinks below comfort.
    w = float(np.clip((float(comfort) - min_sep) / max(float(comfort), 1e-6), 0.0, 1.0))
    avoid = _clip_action((-rel / n) * float(avoid_gain))
    blended = (1.0 - w) * goal.astype(np.float64) + w * avoid.astype(np.float64)
    return _clip_action(blended)


def make_cautious_orca_policy(
    speed: float = 0.7,
    comfort: float = 0.8,
    avoid_gain: float = 1.2,
) -> PolicyFn:
    def _fn(obs: np.ndarray) -> np.ndarray:
        return cautious_orca_policy(
            obs, speed=speed, comfort=comfort, avoid_gain=avoid_gain
        )

    return _fn


def make_go_to_goal_policy(speed: float = 1.0) -> PolicyFn:
    spd = float(speed)

    def _fn(obs: np.ndarray) -> np.ndarray:
        return go_to_goal_policy(obs, speed=spd)

    return _fn


def flee_nearest_human_policy(obs: np.ndarray) -> np.ndarray:
    """Move opposite the nearest-human relative vector (obs[9:11])."""
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    if obs.shape[0] < 11:
        return np.zeros(2, dtype=np.float32)
    rel = obs[9:11]
    n = float(np.linalg.norm(rel))
    if n < 1e-8:
        return np.zeros(2, dtype=np.float32)
    return _clip_action(-rel / n)


def stationary_policy(obs: np.ndarray) -> np.ndarray:
    del obs
    return np.zeros(2, dtype=np.float32)


def random_policy(obs: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    del obs
    rng = rng or np.random.default_rng()
    return _clip_action(rng.uniform(-1.0, 1.0, size=2))


def make_random_policy(seed: int = 0) -> PolicyFn:
    rng = np.random.default_rng(int(seed))

    def _fn(obs: np.ndarray) -> np.ndarray:
        return random_policy(obs, rng=rng)

    return _fn


def make_constant_policy(vx: float = 0.0, vy: float = 0.0) -> PolicyFn:
    action = _clip_action(np.asarray([vx, vy], dtype=np.float32))

    def _fn(obs: np.ndarray) -> np.ndarray:
        del obs
        return action.copy()

    return _fn


def load_sac_policy(model_path: PathLike, device: str = "cpu") -> PolicyFn:
    """Load a Stable-Baselines3 SAC zip; returns deterministic ``predict`` wrapper."""
    from stable_baselines3 import SAC

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"SAC checkpoint not found: {path}")
    model = SAC.load(str(path), device=device)

    def _fn(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
        return _clip_action(action)

    return _fn


def wrap_noisy_policy(
    base: PolicyFn,
    *,
    noise_std: float = 0.3,
    seed: int = 0,
) -> PolicyFn:
    """Add isotropic Gaussian noise to actions, then clip to ``[-1, 1]``."""
    rng = np.random.default_rng(int(seed))
    std = float(noise_std)

    def _fn(obs: np.ndarray) -> np.ndarray:
        action = np.asarray(base(obs), dtype=np.float32).reshape(2)
        if std > 0.0:
            action = action + rng.normal(0.0, std, size=2).astype(np.float32)
        return _clip_action(action)

    return _fn


def get_policy(name: str, **kwargs) -> PolicyFn:
    key = str(name).strip().lower()
    if key in ("go_to_goal", "goto", "greedy"):
        return make_go_to_goal_policy(speed=float(kwargs.get("speed", 1.0)))
    if key in ("slow_go_to_goal", "slow_goto"):
        return make_go_to_goal_policy(speed=float(kwargs.get("speed", 0.35)))
    if key in ("cautious", "cautious_orca", "orca_like"):
        return make_cautious_orca_policy(
            speed=float(kwargs.get("speed", 0.7)),
            comfort=float(kwargs.get("comfort", 0.8)),
            avoid_gain=float(kwargs.get("avoid_gain", 1.2)),
        )
    if key in ("flee", "flee_human", "flee_nearest_human"):
        return flee_nearest_human_policy
    if key in ("noisy_go_to_goal", "noisy_goto"):
        return wrap_noisy_policy(
            make_go_to_goal_policy(speed=1.0),
            noise_std=float(kwargs.get("noise_std", 0.3)),
            seed=int(kwargs.get("seed", 0)),
        )
    if key in ("stationary", "zero", "freeze"):
        return stationary_policy
    if key in ("random",):
        return make_random_policy(seed=int(kwargs.get("seed", 0)))
    if key in ("constant",):
        return make_constant_policy(
            vx=float(kwargs.get("vx", 0.0)),
            vy=float(kwargs.get("vy", 0.0)),
        )
    if key in ("sac", "sb3_sac"):
        return load_sac_policy(kwargs["model_path"], device=str(kwargs.get("device", "cpu")))
    if key in ("noisy_sac", "sac_noisy"):
        base = load_sac_policy(kwargs["model_path"], device=str(kwargs.get("device", "cpu")))
        return wrap_noisy_policy(
            base,
            noise_std=float(kwargs.get("noise_std", 0.3)),
            seed=int(kwargs.get("seed", 0)),
        )
    raise KeyError(f"Unknown policy: {name}")
