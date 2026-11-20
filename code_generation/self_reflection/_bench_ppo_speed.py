"""Quick PPO+GST throughput bench (not part of CARD runtime)."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from code_generation.self_reflection.crowdnav_env_utils import CROWDNAV_ROOT, _ensure_crowdnav_path
from code_generation.self_reflection.crowdnav_pp_env import make_pp_vec_env
from code_generation.self_reflection.crowdnav_ppo_train import default_algo_args


def _run_once(*, device_cuda: bool, torch_threads: int, warmup_updates: int, timed_updates: int) -> Dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    torch.set_num_threads(max(1, torch_threads))
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    prev = os.getcwd()
    _ensure_crowdnav_path()
    os.chdir(CROWDNAV_ROOT)
    try:
        from rl import ppo as ppo_mod  # type: ignore
        from rl.networks.model import Policy  # type: ignore
        from rl.networks.storage import RolloutStorage  # type: ignore

        algo_args = default_algo_args(seed=0, num_env_steps=10_000, device_cuda=device_cuda)
        device = torch.device("cuda:0" if algo_args.cuda else "cpu")
        envs, config, _ = make_pp_vec_env(
            "circle-crossing-v0",
            seed=0,
            use_ground_truth_reward=True,
            reward_code=None,
            max_episode_steps=200,
            device=device,
            restore_llm_reward=False,
        )
        config.training.device = "cuda:0" if device.type == "cuda" else "cpu"

        actor_critic = Policy(
            envs.observation_space.spaces,
            envs.action_space,
            base_kwargs=algo_args,
            base=config.robot.policy,
        )
        nn.DataParallel(actor_critic).to(device)
        agent = ppo_mod.PPO(
            actor_critic,
            algo_args.clip_param,
            algo_args.ppo_epoch,
            algo_args.num_mini_batch,
            algo_args.value_loss_coef,
            algo_args.entropy_coef,
            lr=algo_args.lr,
            eps=algo_args.eps,
            max_grad_norm=algo_args.max_grad_norm,
        )
        rollouts = RolloutStorage(
            algo_args.num_steps,
            algo_args.num_processes,
            envs.observation_space.spaces,
            envs.action_space,
            algo_args.human_node_rnn_size,
            algo_args.human_human_edge_rnn_size,
        )
        obs = envs.reset()
        for key in obs:
            rollouts.obs[key][0].copy_(obs[key])
        rollouts.to(device)

        def one_update() -> None:
            for step in range(algo_args.num_steps):
                with torch.no_grad():
                    rollouts_obs = {k: rollouts.obs[k][step] for k in rollouts.obs}
                    rollouts_hidden = {
                        k: rollouts.recurrent_hidden_states[k][step]
                        for k in rollouts.recurrent_hidden_states
                    }
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts_obs, rollouts_hidden, rollouts.masks[step]
                    )
                if config.env.use_wrapper:
                    out_pred = rollouts_obs["spatial_edges"][:, :, 2:].to("cpu").numpy()
                    envs.talk2Env(out_pred)
                obs, reward, done, infos = envs.step(action)
                del infos
                masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])
                bad_masks = torch.ones(algo_args.num_processes, 1)
                rollouts.insert(
                    obs,
                    recurrent_hidden_states,
                    action,
                    action_log_prob,
                    value,
                    reward,
                    masks,
                    bad_masks,
                )
            with torch.no_grad():
                rollouts_obs = {k: rollouts.obs[k][-1] for k in rollouts.obs}
                rollouts_hidden = {
                    k: rollouts.recurrent_hidden_states[k][-1]
                    for k in rollouts.recurrent_hidden_states
                }
                next_value = actor_critic.get_value(
                    rollouts_obs, rollouts_hidden, rollouts.masks[-1]
                ).detach()
            rollouts.compute_returns(
                next_value,
                algo_args.use_gae,
                algo_args.gamma,
                algo_args.gae_lambda,
                algo_args.use_proper_time_limits,
            )
            agent.update(rollouts)
            rollouts.after_update()

        for _ in range(warmup_updates):
            one_update()
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(timed_updates):
            one_update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        steps = timed_updates * algo_args.num_steps * algo_args.num_processes
        envs.close()
        return {
            "device": str(device),
            "torch_threads": torch_threads,
            "updates": timed_updates,
            "env_steps": steps,
            "seconds": round(dt, 3),
            "steps_per_sec": round(steps / dt, 2),
        }
    finally:
        os.chdir(prev)


def main() -> None:
    configs: List[Dict[str, Any]] = []
    thread_grid = [1, 2, 4, 8]
    devices = [True]
    if torch.cuda.is_available():
        devices = [True, False]  # cuda then cpu
    for use_cuda in devices:
        for th in thread_grid:
            label = f"{'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'} threads={th}"
            print(f"[bench] {label} ...", flush=True)
            try:
                row = _run_once(
                    device_cuda=use_cuda,
                    torch_threads=th,
                    warmup_updates=2,
                    timed_updates=8,  # 8*30=240 env steps timed
                )
                configs.append(row)
                print(
                    f"  -> {row['steps_per_sec']} steps/s "
                    f"({row['env_steps']} steps in {row['seconds']}s)",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL: {exc}", flush=True)

    if not configs:
        print("No successful benches")
        return
    best = max(configs, key=lambda r: r["steps_per_sec"])
    print("\n=== BEST ===")
    print(best)

    # Current script budget: 3*150k + 2*(1M-150k) = 450k + 1.7M = 2.15M
    total_rl = 3 * 150_000 + 2 * (1_000_000 - 150_000)
    # CARD also does evals / LLM; RL-only ETA first, then +30% overhead estimate.
    sps = float(best["steps_per_sec"])
    hours_rl = total_rl / sps / 3600.0
    hours_wall = hours_rl * 1.35  # eval + LLM + gate overhead rough
    print(f"\nBudget env-steps ≈ {total_rl:,}")
    print(f"RL-only ETA @ {sps} steps/s ≈ {hours_rl:.1f} h")
    print(f"Wall ETA (+~35% eval/LLM) ≈ {hours_wall:.1f} h  ({hours_wall/24:.1f} days)")


if __name__ == "__main__":
    main()
