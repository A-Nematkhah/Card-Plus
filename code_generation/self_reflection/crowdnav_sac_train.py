"""SAC + flat CrowdNavEnv training (legacy CARD stack that hit ~92% @ H=10)."""

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

from code_generation.self_reflection.crowdnav_env_utils import (
    CROWDNAV_TASKS,
    load_reward_function,
    make_env,
)
from code_generation.self_reflection.crowdnav_exp import mapping_dicts
from code_generation.self_reflection.utils import (
    SubRewardLog,
    collect_trajectories,
    print_with_color,
    set_seed_everywhere,
)

try:
    import wandb
except ImportError:  # optional
    wandb = None

try:
    from wandb.integration.sb3 import WandbCallback
except ImportError:  # optional
    WandbCallback = None  # type: ignore[misc, assignment]


def train_and_eval_sac(
    args: Any,
    seed: int,
    step: int,
    use_ground_truth_reward: bool,
    specific_code: str,
    specific_code_path: str,
    load_model: bool,
    *,
    resume_model_path: Optional[str] = None,
    train_steps: Optional[int] = None,
) -> Tuple[dict, list, list]:
    """Train (or continue) SAC on flat obs and collect eval trajectories."""
    log_path_root = os.path.join(os.path.dirname(specific_code_path), f"seed_{seed}")
    os.makedirs(log_path_root, exist_ok=True)
    if not use_ground_truth_reward:
        load_reward_function(code=specific_code)
    steps_budget = int(args.train_max_steps if train_steps is None else train_steps)

    if args.task_name not in CROWDNAV_TASKS:
        raise ValueError(f"Unknown CrowdNav task: {args.task_name}")

    run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed; pip install wandb or drop --use_wandb")
        run = wandb.init(
            project=args.benchmark_name,
            entity=args.entity,
            dir=log_path_root,
            group=args.task_name,
            config=vars(args),
            name="-".join(
                [
                    args.llm_model_name,
                    "zeroshot",
                    args.task_name,
                    args.exp_name,
                    f"step_{step}",
                    f"seed_{seed}",
                ]
            ),
            mode="offline" if args.offline else "online",
            sync_tensorboard=True,
            save_code=True,
        )
        if specific_code_path is not None and glob.glob(
            os.path.join(os.path.dirname(specific_code_path), "*.py")
        ):
            dest = Path(wandb.run.dir) / "reward_codes" / run.id
            dest.mkdir(parents=True, exist_ok=True)
            for src in Path(specific_code_path).parent.glob("*.py"):
                shutil.copy2(src, dest / src.name)

    vec_choice = str(getattr(args, "vec_env", "auto")).lower()
    if vec_choice == "auto":
        TrainVecCls = SubprocVecEnv if args.train_num > 1 else DummyVecEnv
    elif vec_choice == "subproc":
        TrainVecCls = SubprocVecEnv
    else:
        TrainVecCls = DummyVecEnv
    EvalVecCls = DummyVecEnv
    print_with_color(
        f"CrowdNav train backend: SAC + flat obs "
        f"(train={TrainVecCls.__name__} n={args.train_num})",
        "BLUE",
    )

    reward_code = None if use_ground_truth_reward else specific_code
    eval_env = EvalVecCls(
        [
            make_env(
                args,
                seed=seed,
                record_dir="logs/videos",
                is_eval=True,
                use_ground_truth_reward=use_ground_truth_reward,
                reward_code=reward_code,
            )
            for _ in range(args.eval_num)
        ]
    )
    eval_env = VecMonitor(eval_env)
    eval_env.seed(seed)
    eval_env.reset()

    extra_eval_env = make_env(
        args,
        seed=seed,
        record_dir="logs/videos",
        is_eval=True,
        use_ground_truth_reward=use_ground_truth_reward,
        reward_code=reward_code,
    )()
    extra_eval_env.seed(seed)
    extra_eval_env.reset()

    env = TrainVecCls(
        [
            make_env(
                args,
                seed=seed,
                max_episode_steps=args.max_episode_steps,
                is_eval=False,
                use_ground_truth_reward=use_ground_truth_reward,
                reward_code=reward_code,
            )
            for _ in range(args.train_num)
        ]
    )
    env = VecMonitor(env)
    # Soft scale handling: keep LLM magnitudes unconstrained, but whiten rewards
    # for SAC so gate-favored huge collision terminals do not dominate gradients.
    norm_reward = os.environ.get("CARD_SAC_NORM_REWARD", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if norm_reward:
        env = VecNormalize(
            env,
            training=True,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
            gamma=0.99,
        )
        print_with_color(
            "SAC VecNormalize(norm_reward=True, clip_reward=10) enabled "
            "(set CARD_SAC_NORM_REWARD=0 to disable)",
            "BLUE",
        )
    env.seed(seed)
    env.reset()

    eval_log_path = os.path.join(log_path_root, "tensorboard_log/eval")
    os.makedirs(eval_log_path, exist_ok=True)
    step_eval_freq = max(1, args.eval_freq // args.train_num)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=eval_log_path,
        log_path=eval_log_path,
        eval_freq=step_eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=args.n_eval_episodes,
    )
    log_file_name = "extra_info.npz"
    sub_eval_mult = max(1, int(getattr(args, "sub_reward_eval_mult", 2)))
    extra_info_callback = SubRewardLog(
        eval_env=extra_eval_env,
        eval_freq=step_eval_freq * sub_eval_mult,
        log_path=eval_log_path,
        log_file_name=log_file_name,
        n_eval_episodes=args.n_eval_episodes,
        verbose=1,
    )
    using_cuda = torch.cuda.is_available()
    set_seed_everywhere(seed, using_cuda)
    torch.set_num_threads(max(1, int(getattr(args, "torch_threads", 1))))

    train_log_path = os.path.join(log_path_root, "tensorboard_log")
    os.makedirs(train_log_path, exist_ok=True)
    callback_list = [eval_callback, extra_info_callback]
    if args.use_wandb and WandbCallback is not None:
        callback_list.append(WandbCallback(verbose=2))

    def _save_policy() -> None:
        model.save(os.path.join(train_log_path, "latest_model"))
        if norm_reward and isinstance(env, VecNormalize):
            env.save(os.path.join(train_log_path, "vecnormalize.pkl"))

    def _maybe_load_vecnorm() -> None:
        nonlocal env
        if not norm_reward:
            return
        stats = os.path.join(train_log_path, "vecnormalize.pkl")
        if os.path.isfile(stats) and isinstance(env, VecNormalize):
            env = VecNormalize.load(stats, env)
            env.training = True
            env.norm_reward = True

    if load_model and not resume_model_path:
        resume_path = os.path.join(train_log_path, "latest_model.zip")
        _maybe_load_vecnorm()
        model = SAC.load(resume_path, env=env)
    elif resume_model_path:
        print_with_color(
            f"Continuing SAC from {resume_model_path} for +{steps_budget} steps",
            "BLUE",
        )
        _maybe_load_vecnorm()
        model = SAC.load(resume_model_path, env=env)
        model.set_logger(configure(train_log_path, ["stdout", "csv", "tensorboard"]))
        try:
            model.learn(
                total_timesteps=steps_budget,
                callback=callback_list,
                reset_num_timesteps=False,
            )
        finally:
            _save_policy()
    else:
        policy_kwargs = dict(net_arch=[256, 256])
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            batch_size=256,
            gamma=0.99,
            learning_rate=0.0003,
            tau=0.005,
            learning_starts=1000,
            ent_coef="auto_0.1",
            seed=seed,
        )
        model.set_logger(configure(train_log_path, ["stdout", "csv", "tensorboard"]))
        try:
            model.learn(steps_budget, callback=callback_list)
        finally:
            _save_policy()

    learning_result = np.load(os.path.join(eval_log_path, "evaluations.npz"))
    learning_result = {key: learning_result[key] for key in learning_result.files}
    extra_info_path = os.path.join(eval_log_path, log_file_name)
    if os.path.isfile(extra_info_path):
        extra_info = np.load(extra_info_path)
        learning_result.update({key: extra_info[key] for key in extra_info.files})

    check_traj_env = make_env(
        args,
        seed=seed,
        record_dir="logs/videos",
        is_eval=True,
        use_ground_truth_reward=use_ground_truth_reward,
        reward_code=reward_code,
    )()
    check_traj_env.seed(seed)
    check_traj_env.reset()
    diff_len_env = make_env(
        args,
        seed=seed,
        record_dir="logs/videos",
        is_eval=True,
        use_ground_truth_reward=use_ground_truth_reward,
        reward_code=reward_code,
    )()
    diff_len_env.seed(seed)
    diff_len_env.reset()
    trajectories_for_check = collect_trajectories(
        model=model,
        env=check_traj_env,
        mapping_dicts=mapping_dicts,
        trajectory_num=args.trajectory_num,
        benchmark_name=args.benchmark_name,
    )
    diff_len_trajectories = collect_trajectories(
        model=model,
        env=diff_len_env,
        mapping_dicts=mapping_dicts,
        trajectory_num=args.n_eval_episodes,
        benchmark_name=args.benchmark_name,
    )
    if args.use_wandb and run is not None:
        run.finish()
        wandb.finish()
    return learning_result, trajectories_for_check, diff_len_trajectories
