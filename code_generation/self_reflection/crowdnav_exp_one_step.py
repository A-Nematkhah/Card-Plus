import argparse
import glob
import json
import os
import pickle
import sys
import traceback
from datetime import datetime
from typing import Dict, Tuple

import gym
import numpy as np
import torch
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from wandb.integration.sb3 import WandbCallback

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from code_generation.self_reflection.crowdnav_env_utils import (
    CROWDNAV_TASKS,
    ContinuousTaskWrapper,
    load_reward_function,
    make_env,
)
from code_generation.self_reflection.crowdnav_exp import mapping_dicts
from card_plus.archive import RewardArchive
from card_plus.baseline_feedback import assemble_baseline_feedback
from card_plus.config import add_card_plus_arguments, card_plus_from_args
from card_plus.dataset.gate import (
    add_dataset_gate_arguments,
    dataset_gate_from_args,
    run_dataset_gate,
    should_skip_rl_for_dataset_gate,
)
from card_plus.dataset.reward_static import check_reward_code_static
from card_plus.hooks import (
    apply_card_plus_after_baseline,
    run_optional_tpe_v2,
)
from card_plus.tpe_v2 import should_skip_rl_for_tpe_v2
from code_generation.self_reflection.utils import (
    Logger,
    SubRewardLog,
    collect_trajectories,
    convert_function_external,
    get_eval_feedback,
    get_learning_result_feedback_simple,
    get_learning_result_mean,
    print_with_color,
    set_seed_everywhere,
    trajectories_return_check,
)

LLM_RESPONSE = "llm_response_code"
RL_QUERY = "rl_query_code"
CHECKPOINT = "checkpoint"


def env_is_valid(reward_code_str, args):
    static_ok, static_reasons = check_reward_code_static(reward_code_str)
    if not static_ok:
        return False, "static reward check failed: " + "; ".join(static_reasons)
    try:
        train_env = make_env(
            args,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            is_eval=False,
            use_ground_truth_reward=False,
            reward_code=reward_code_str,
        )()
        eval_env = make_env(
            args,
            seed=args.seed,
            record_dir="logs/videos",
            is_eval=True,
            use_ground_truth_reward=False,
            reward_code=reward_code_str,
        )()
        external_reward_code = convert_function_external(reward_code_str, benchmark_name=args.benchmark_name)
        for env in (train_env, eval_env):
            obs = env.reset()
            action = env.action_space.sample()
            load_reward_function(code=reward_code_str)
            obs, reward, done, info = env.step(action)
            assert isinstance(reward, (float, np.floating)), "reward is not float type"

            load_reward_function(code=external_reward_code)
            external_reward_func = ContinuousTaskWrapper.compute_dense_reward
            # CrowdNav external form uses pre_obs (former self.pre_obs) for delta progress.
            result = external_reward_func(action=action, obs=obs, pre_obs=obs)
            assert isinstance(result, tuple), "external_reward_func return type is not tuple"
    except Exception:
        return False, traceback.format_exc()
    return True, ""


def train_and_eval(args, seed, step, use_ground_truth_reward: bool, specific_code, specific_code_path: str, load_model):
    log_path_root = os.path.join(os.path.dirname(specific_code_path), f"seed_{seed}")
    os.makedirs(log_path_root, exist_ok=True)
    if not use_ground_truth_reward:
        load_reward_function(code=specific_code)

    if args.task_name not in CROWDNAV_TASKS:
        raise ValueError(f"Unknown CrowdNav task: {args.task_name}")

    if args.use_wandb:
        run = wandb.init(
            project=args.benchmark_name,
            entity=args.entity,
            dir=log_path_root,
            group=args.task_name,
            config=vars(args),
            name="-".join([args.llm_model_name, "zeroshot", args.task_name, args.exp_name, f"step_{step}", f"seed_{seed}"]),
            mode="offline" if args.offline else "online",
            sync_tensorboard=True,
            save_code=True,
        )
        if specific_code_path is not None and glob.glob(os.path.join(os.path.dirname(specific_code_path), "*.py")):
            os.makedirs(f"{wandb.run.dir}/reward_codes/{run.id}", exist_ok=True)
            os.system(f"cp -r {os.path.dirname(specific_code_path)}/*.py {wandb.run.dir}/reward_codes/{run.id}")

    # DummyVecEnv is serial (no FPS gain with more envs). SubprocVecEnv parallelizes ORCA.
    # Keep eval on Dummy to avoid doubling torch-worker RAM on Windows.
    vec_choice = str(getattr(args, "vec_env", "auto")).lower()
    if vec_choice == "auto":
        TrainVecCls = SubprocVecEnv if args.train_num > 1 else DummyVecEnv
    elif vec_choice == "subproc":
        TrainVecCls = SubprocVecEnv
    else:
        TrainVecCls = DummyVecEnv
    EvalVecCls = DummyVecEnv
    print(
        f"Using train={TrainVecCls.__name__}(n={args.train_num}) "
        f"eval={EvalVecCls.__name__}(n={args.eval_num})"
    )

    reward_code = None if use_ground_truth_reward else specific_code
    eval_env = EvalVecCls([
        make_env(
            args,
            seed=seed,
            record_dir="logs/videos",
            is_eval=True,
            use_ground_truth_reward=use_ground_truth_reward,
            reward_code=reward_code,
        )
        for _ in range(args.eval_num)
    ])
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

    env = TrainVecCls([
        make_env(
            args,
            seed=seed,
            max_episode_steps=args.max_episode_steps,
            is_eval=False,
            use_ground_truth_reward=use_ground_truth_reward,
            reward_code=reward_code,
        )
        for _ in range(args.train_num)
    ])
    env = VecMonitor(env)
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
    # SubRewardLog is single-env serial; run less often so it does not dominate wall FPS.
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
    # Env stepping is CPU-bound (ORCA); keep torch from oversubscribing cores.
    torch.set_num_threads(max(1, int(getattr(args, "torch_threads", 1))))

    if load_model:
        resume_path = os.path.join(log_path_root, "tensorboard_log", "latest_model.zip")
        model = SAC.load(resume_path, env=env)
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
        train_log_path = os.path.join(log_path_root, "tensorboard_log")
        os.makedirs(train_log_path, exist_ok=True)
        model.set_logger(configure(train_log_path, ["stdout", "csv", "tensorboard"]))
        callback_list = [eval_callback, extra_info_callback]
        if args.use_wandb:
            callback_list.append(WandbCallback(verbose=2))
        try:
            model.learn(args.train_max_steps, callback=callback_list)
        finally:
            # Persist the policy even if a logging callback crashes after the last step.
            model.save(os.path.join(train_log_path, "latest_model"))

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
        model=model, env=check_traj_env, mapping_dicts=mapping_dicts,
        trajectory_num=args.trajectory_num, benchmark_name=args.benchmark_name,
    )
    diff_len_trajectories = collect_trajectories(
        model=model, env=diff_len_env, mapping_dicts=mapping_dicts,
        trajectory_num=args.n_eval_episodes, benchmark_name=args.benchmark_name,
    )
    if args.use_wandb:
        run.finish()
        wandb.finish()
    return learning_result, trajectories_for_check, diff_len_trajectories


def load_llm_response(args, llm_response_file_name):
    llm_ans_file = os.path.join(args.log_path, llm_response_file_name)
    try:
        with open(llm_ans_file, "rb") as file:
            loaded_objects_dict = pickle.load(file)
    except Exception as exc:
        print(f"{exc}")
        return False, {}
    return True, loaded_objects_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="card")
    parser.add_argument("--task_name", type=str, default="circle-crossing-v0")
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--llm_model_name", type=str, default="grok-4.3")
    parser.add_argument("--reasoning_effort", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--reflection_num", type=int, default=10)
    parser.add_argument("--checkpoint", action="store_true", default=False)
    parser.add_argument("--load_model", action="store_true", default=False)
    parser.add_argument("--train_num", type=int, default=4)
    parser.add_argument("--eval_num", type=int, default=4)
    parser.add_argument("--vec_env", type=str, default="auto", choices=["auto", "dummy", "subproc"],
                        help="auto: SubprocVecEnv when train_num>1 (needed for FPS scaling on Windows)")
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--sub_reward_eval_mult", type=int, default=2,
                        help="Run SubRewardLog every N eval periods (higher => higher train FPS)")
    parser.add_argument("--eval_freq", type=int, default=5_000)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--train_max_steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--benchmark_name", type=str, default="crowdnav")
    parser.add_argument("--max_try_num", type=int, default=10)
    parser.add_argument("--baseline_reward_path", type=str, default="")
    parser.add_argument("--trajectory_num", type=int, default=40)
    parser.add_argument("--train_seed_num", type=int, default=1)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--use_return_check", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--log_point_num", type=int, default=4)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--entity", type=str, default="Automated-Reward-Generation-with-Introspection")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--step", type=int)
    parser.add_argument("--exclude", type=str, default=None)
    parser.add_argument("--query_restart", type=bool, default=True)
    add_card_plus_arguments(parser)
    add_dataset_gate_arguments(parser)
    args = parser.parse_args()
    card_plus_cfg = card_plus_from_args(args)
    dataset_gate_cfg = dataset_gate_from_args(args)

    if args.log_path is None:
        formatted_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        args.log_path = (
            f"logs/{args.benchmark_name}/{args.llm_model_name}/t={args.temperature}/zero-shot/"
            f"SAC_{args.seed}/{args.task_name}/{args.exp_name}/exclude={args.exclude}/{formatted_time}"
        )
    os.makedirs(args.log_path, exist_ok=True)

    if args.exp_name == "card":
        std_file = os.path.join(args.log_path, f"{RL_QUERY}_{args.step}.log")
        logger = Logger(filename=std_file)
        sys.stdout = logger
        sys.stderr = logger
        print("args is:")
        print(json.dumps(vars(args), indent=4))
        if args.step == 0:
            query_file_path = os.path.join(args.log_path, f"{RL_QUERY}_{args.step}.pkl")
            with open(query_file_path, "wb") as file:
                pickle.dump(
                    {
                        "rl_args": args,
                        "step": 0,
                        "message_history": [],
                        "trajectories_for_check_list": [],
                        "feedback_str": "",
                    },
                    file,
                )
        else:
            code_file = f"{LLM_RESPONSE}_{args.step - 1}.pkl"
            load_success, loaded_objects_dict = load_llm_response(args=args, llm_response_file_name=code_file)
            if not load_success:
                exit(0)
            general_code_path = os.path.join(args.log_path, f"code_{args.step - 1}", f"general_{args.step - 1}.py")
            specific_code_path = os.path.join(args.log_path, f"code_{args.step - 1}", f"specific_{args.step - 1}.py")
            with open(general_code_path, "r", encoding="utf-8") as f:
                general_code = f.read()
            with open(specific_code_path, "r", encoding="utf-8") as f:
                specific_code = f.read()
            message_history = loaded_objects_dict["message_history"]
            trajectories_for_check_list = loaded_objects_dict["trajectories_for_check_list"]

            feedback_str = ""
            correct_count = 0
            correct_all = 0
            correct_rate = 1
            pass_flag = True
            check_feedback = ""
            tpe_v2_result = None
            dataset_gate_result = None
            rl_ran = True
            reward_function = None
            archive = RewardArchive(run_dir=args.log_path, cfg=card_plus_cfg) if card_plus_cfg.any_enabled else None
            if args.checkpoint:
                checkpoint_file = os.path.join(args.log_path, f"{CHECKPOINT}_{args.step}.pkl")
                with open(checkpoint_file, "rb") as f:
                    loaded_data = pickle.load(f)
                correct_count, correct_all, correct_rate, pass_flag, check_feedback = loaded_data["return_check_info"]
                learning_result, trajectories_for_check_need_update, diff_len_trajectories = loaded_data["train_info"]
                tpe_v2_result = loaded_data.get("tpe_v2_result")
                rl_ran = loaded_data.get("rl_ran", learning_result is not None)
                print("=" * 20 + f"{args.exp_name} load checkpoint successfully!" + "=" * 20)
                print(f"checkpoint path is: {checkpoint_file}" + "=" * 20)
            else:
                if dataset_gate_cfg.enabled:
                    try:
                        dataset_gate_result = run_dataset_gate(
                            dataset_gate_cfg,
                            reward_code=specific_code,
                            run_dir=args.log_path,
                            step=args.step,
                        )
                    except Exception as exc:
                        print_with_color(f"dataset gate failed (continuing without it): {exc}", "YELLOW")
                        dataset_gate_result = None
                if args.use_return_check and trajectories_for_check_list:
                    external_call_reward_code = convert_function_external(
                        reward_function_code=specific_code, benchmark_name=args.benchmark_name
                    )
                    load_reward_function(code=external_call_reward_code)
                    reward_function = ContinuousTaskWrapper.compute_dense_reward
                    correct_count, correct_all, correct_rate, pass_flag, check_feedback = trajectories_return_check(
                        reward_function=reward_function,
                        trajectories=trajectories_for_check_list,
                        mapping_dicts=mapping_dicts,
                        threshold=args.threshold,
                        log_interval=args.log_interval,
                        benchmark_name=args.benchmark_name,
                        env=None,
                    )
                    if card_plus_cfg.use_tpe_v2:
                        tpe_v2_result = run_optional_tpe_v2(
                            card_plus_cfg,
                            trajectories_for_check_list,
                            pass_v1=pass_flag,
                            acc_v1=float(correct_rate),
                            task_name=args.task_name,
                            reward_fn=reward_function,
                            benchmark_name=args.benchmark_name,
                        )

                skip_rl = bool(
                    card_plus_cfg.use_tpe_v2
                    and tpe_v2_result is not None
                    and should_skip_rl_for_tpe_v2(card_plus_cfg, tpe_v2_result)
                )
                if should_skip_rl_for_dataset_gate(dataset_gate_result):
                    skip_rl = True
                    print(
                        "dataset_gate: skipping RL "
                        f"(composite={dataset_gate_result.composite:.4f}; "
                        f"{', '.join(dataset_gate_result.reasons) or 'keep=False'})"
                    )
                if skip_rl:
                    if (
                        card_plus_cfg.use_tpe_v2
                        and tpe_v2_result is not None
                        and should_skip_rl_for_tpe_v2(card_plus_cfg, tpe_v2_result)
                    ):
                        print(
                            "CARD+ TPE v2 gate: dense-shaping check failed (hacked); "
                            "skipping RL training for this version."
                        )
                    learning_result, trajectories_for_check_need_update, diff_len_trajectories = None, [], []
                    rl_ran = False
                else:
                    learning_result, trajectories_for_check_need_update, diff_len_trajectories = train_and_eval(
                        args=args,
                        seed=args.seed,
                        step=args.step,
                        use_ground_truth_reward=False,
                        specific_code=specific_code,
                        specific_code_path=specific_code_path,
                        load_model=args.load_model,
                    )
                    rl_ran = True
                checkpoint_file = os.path.join(args.log_path, f"{CHECKPOINT}_{args.step}.pkl")
                with open(checkpoint_file, "wb") as f:
                    pickle.dump(
                        {
                            "return_check_info": [correct_count, correct_all, correct_rate, pass_flag, check_feedback],
                            "train_info": [learning_result, trajectories_for_check_need_update, diff_len_trajectories],
                            "tpe_v2_result": tpe_v2_result,
                            "rl_ran": rl_ran,
                        },
                        f,
                    )
                    print("=" * 20 + f"{args.exp_name} checkpoint saved!" + "=" * 20)
                    print(f"load path is: {checkpoint_file}" + "=" * 20)

                if rl_ran and args.train_seed_num > 1:
                    learning_result_list = [learning_result]
                    for seed in range(args.seed + 1, args.seed + args.train_seed_num):
                        learning_result_more, _, _ = train_and_eval(
                            args=args,
                            seed=seed,
                            step=args.step,
                            use_ground_truth_reward=False,
                            specific_code=specific_code,
                            specific_code_path=specific_code_path,
                            load_model=args.load_model,
                        )
                        learning_result_list.append(learning_result_more)
                    get_learning_result_mean(
                        learning_result_list=learning_result_list,
                        save_path=os.path.join(os.path.dirname(specific_code_path), "evaluations_mean.npz"),
                    )

            effective_pass = pass_flag
            if (not rl_ran) and card_plus_cfg.use_tpe_v2 and card_plus_cfg.tpe_v2_mode == "gate":
                effective_pass = False
            if should_skip_rl_for_dataset_gate(dataset_gate_result):
                effective_pass = False

            # Feedback must never discard a finished train/checkpoint; always emit rl_query pkl.
            learning_result_feedback = ""
            eval_feedback = ""
            if rl_ran and learning_result is not None:
                try:
                    learning_result_feedback = get_learning_result_feedback_simple(
                        learning_result=learning_result, log_point_num=args.log_point_num
                    )
                except Exception as exc:
                    learning_result_feedback = f"learning-result feedback unavailable: {exc}\n"
                    print_with_color(learning_result_feedback, "YELLOW")
                try:
                    eval_feedback = get_eval_feedback(
                        trajectory_list=diff_len_trajectories,
                        mapping_dicts=mapping_dicts,
                        log_interval=args.log_interval,
                        benchmark_name=args.benchmark_name,
                    )
                except Exception as exc:
                    eval_feedback = f"trajectory feedback unavailable: {exc}\n"
                    print_with_color(eval_feedback, "YELLOW")
            try:
                feedback_str, extend_pool = assemble_baseline_feedback(
                    use_return_check=args.use_return_check,
                    pass_flag=effective_pass,
                    exclude=args.exclude,
                    learning_result_feedback=learning_result_feedback,
                    eval_feedback=eval_feedback,
                    check_feedback=check_feedback,
                )
                if not args.use_return_check:
                    with open(os.path.join(os.path.dirname(specific_code_path), "no check.txt"), "w") as file:
                        file.write("no check")
                elif effective_pass and rl_ran:
                    with open(os.path.join(os.path.dirname(specific_code_path), "check pass.txt"), "w") as file:
                        file.write("check pass")
                elif not effective_pass:
                    fail_msg = "check fail"
                    if should_skip_rl_for_dataset_gate(dataset_gate_result):
                        fail_msg = "dataset gate fail"
                    elif not (pass_flag or rl_ran):
                        fail_msg = "tpe_v2 gate fail"
                    with open(os.path.join(os.path.dirname(specific_code_path), "check fail.txt"), "w") as file:
                        file.write(fail_msg)
                if extend_pool and rl_ran:
                    trajectories_for_check_list.extend(trajectories_for_check_need_update)
            except Exception as exc:
                feedback_str += learning_result_feedback + "\n" + eval_feedback
                print_with_color(f"feedback assembly fallback used: {exc}", "YELLOW")

            if dataset_gate_result is not None:
                feedback_str = feedback_str + "\n" + dataset_gate_result.feedback_text()

            if card_plus_cfg.any_enabled:
                stats_trajs = (
                    trajectories_for_check_need_update
                    if rl_ran and trajectories_for_check_need_update
                    else trajectories_for_check_list
                )
                if reward_function is None and args.use_return_check:
                    external_call_reward_code = convert_function_external(
                        reward_function_code=specific_code, benchmark_name=args.benchmark_name
                    )
                    load_reward_function(code=external_call_reward_code)
                    reward_function = ContinuousTaskWrapper.compute_dense_reward

                def _train_final(code: str):
                    print(
                        "CARD+ score_final_version: extra short-protocol training "
                        f"for unscored last version (step={args.step})."
                    )
                    lr, _, _ = train_and_eval(
                        args=args,
                        seed=args.seed,
                        step=args.step,
                        use_ground_truth_reward=False,
                        specific_code=code,
                        specific_code_path=specific_code_path,
                        load_model=False,
                    )
                    return lr

                feedback_str = apply_card_plus_after_baseline(
                    feedback_str,
                    card_plus_cfg,
                    archive=archive,
                    iteration=args.step,
                    reward_code=specific_code,
                    trajs=stats_trajs,
                    task_name=args.task_name,
                    benchmark_name=args.benchmark_name,
                    tpe_v2=tpe_v2_result,
                    rl_ran=rl_ran,
                    learning_result=learning_result,
                    pass_v1=tpe_v2_result.pass_v1 if tpe_v2_result is not None else pass_flag,
                    acc_v1=float(correct_rate),
                    run_dir=args.log_path,
                    reward_fn=reward_function,
                    train_final_fn=_train_final,
                )

            query_file_path = os.path.join(args.log_path, f"{RL_QUERY}_{args.step}.pkl")
            with open(query_file_path, "wb") as file:
                pickle.dump(
                    {
                        "rl_args": args,
                        "step": args.step,
                        "message_history": message_history,
                        "trajectories_for_check_list": trajectories_for_check_list,
                        "feedback_str": feedback_str,
                    },
                    file,
                )
            print(f"wrote query file: {query_file_path}")

    print(args.exp_name, "finished!")
