"""Short SAC rungs for CrowdNav successive-halving candidate selection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from code_generation.self_reflection.crowdnav_env_utils import load_reward_function, make_env
from code_generation.self_reflection.utils import print_with_color, set_seed_everywhere


def _mean_last_success(eval_npz: Path) -> float:
    """Read last EvalCallback success rate; 0.0 if missing."""
    path = Path(eval_npz) / "evaluations.npz"
    if not path.is_file():
        return 0.0
    data = np.load(path)
    if "successes" not in data.files:
        return 0.0
    successes = np.asarray(data["successes"], dtype=float)
    if successes.size == 0:
        return 0.0
    last = successes[-1]
    return float(np.mean(last))


def train_candidate_rung(
    args: Any,
    *,
    reward_code: str,
    steps: int,
    work_dir: str,
    resume_path: Optional[str] = None,
    seed: Optional[int] = None,
    n_eval_episodes: Optional[int] = None,
) -> Tuple[float, str]:
    """Train (or continue) SAC for ``steps``; return (success_rate, model_zip)."""
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")

    work_dir_p = Path(work_dir)
    work_dir_p.mkdir(parents=True, exist_ok=True)
    tb_dir = work_dir_p / "tensorboard_log"
    eval_dir = tb_dir / "eval"
    tb_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    model_path = str(tb_dir / "latest_model")

    seed = int(args.seed if seed is None else seed)
    n_eval = int(
        n_eval_episodes
        if n_eval_episodes is not None
        else getattr(args, "sh_eval_episodes", 20)
    )
    # Keep SH light and serial — selection fidelity matters more than FPS.
    train_n = max(1, min(2, int(getattr(args, "train_num", 1))))
    eval_n = max(1, min(2, int(getattr(args, "eval_num", 1))))

    load_reward_function(code=reward_code)
    train_env = DummyVecEnv(
        [
            make_env(
                args,
                seed=seed + i,
                max_episode_steps=args.max_episode_steps,
                is_eval=False,
                use_ground_truth_reward=False,
                reward_code=reward_code,
            )
            for i in range(train_n)
        ]
    )
    train_env = VecMonitor(train_env)
    train_env.seed(seed)
    train_env.reset()

    eval_env = DummyVecEnv(
        [
            make_env(
                args,
                seed=seed + 100 + i,
                record_dir="logs/videos",
                is_eval=True,
                use_ground_truth_reward=False,
                reward_code=reward_code,
            )
            for i in range(eval_n)
        ]
    )
    eval_env = VecMonitor(eval_env)
    eval_env.seed(seed + 100)
    eval_env.reset()

    using_cuda = torch.cuda.is_available()
    set_seed_everywhere(seed, using_cuda)
    torch.set_num_threads(max(1, int(getattr(args, "torch_threads", 1))))

    eval_freq = max(1, int(getattr(args, "eval_freq", 5_000)) // train_n)
    # Ensure at least one eval near the end of a short rung.
    eval_freq = min(eval_freq, max(1, steps // train_n))

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(eval_dir),
        log_path=str(eval_dir),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=n_eval,
    )

    try:
        if resume_path and os.path.isfile(resume_path):
            print_with_color(f"SH resume {resume_path} +{steps} steps", "BLUE")
            model = SAC.load(resume_path, env=train_env)
        elif resume_path and os.path.isfile(resume_path + ".zip"):
            print_with_color(f"SH resume {resume_path}.zip +{steps} steps", "BLUE")
            model = SAC.load(resume_path, env=train_env)
        else:
            print_with_color(f"SH fresh train {steps} steps in {work_dir_p}", "BLUE")
            model = SAC(
                "MlpPolicy",
                train_env,
                policy_kwargs=dict(net_arch=[256, 256]),
                verbose=0,
                batch_size=256,
                gamma=0.99,
                learning_rate=0.0003,
                tau=0.005,
                learning_starts=min(1000, max(100, steps // 10)),
                ent_coef="auto_0.1",
                seed=seed,
            )
            model.set_logger(configure(str(tb_dir), ["stdout", "csv"]))

        model.learn(total_timesteps=int(steps), callback=eval_cb, reset_num_timesteps=False)
        model.save(model_path)
    finally:
        try:
            train_env.close()
        except Exception:
            pass
        try:
            eval_env.close()
        except Exception:
            pass

    success = _mean_last_success(eval_dir)
    # Force a final eval file touch: if EvalCallback never wrote, success stays 0.
    return success, model_path + ".zip" if os.path.isfile(model_path + ".zip") else model_path
