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


def go_to_goal_policy(obs: np.ndarray) -> np.ndarray:
    """Unit direction toward goal in obs frame (obs[0:2] pos, obs[4:6] goal)."""
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    delta = obs[4:6] - obs[0:2]
    n = float(np.linalg.norm(delta))
    if n < 1e-8:
        return np.zeros(2, dtype=np.float32)
    return _clip_action(delta / n)


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
        return go_to_goal_policy
    if key in ("noisy_go_to_goal", "noisy_goto"):
        return wrap_noisy_policy(
            go_to_goal_policy,
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
