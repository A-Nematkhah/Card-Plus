"""Profile CrowdNav PPO+GST: CPU vs CUDA and env/policy time split.

Architecture note:
  - ORCA / gym step always runs on CPU.
  - With CARD_DEVICE=cuda, policy + GST tensors use GPU (natural hybrid).
  - There is no practical 'full GPU' for this stack; 'cuda' == hybrid.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn

from code_generation.self_reflection.crowdnav_env_utils import CROWDNAV_ROOT, _ensure_crowdnav_path
from code_generation.self_reflection.crowdnav_pp_env import make_pp_vec_env
from code_generation.self_reflection.crowdnav_ppo_train import default_algo_args


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _profile(*, use_cuda: bool, torch_threads: int = 4, warmup: int = 2, timed: int = 6) -> Dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    os.environ["CARD_DEVICE"] = "cuda" if use_cuda else "cpu"
    torch.set_num_threads(max(1, torch_threads))

    prev = os.getcwd()
    _ensure_crowdnav_path()
    os.chdir(CROWDNAV_ROOT)
    try:
        from rl import ppo as ppo_mod  # type: ignore
        from rl.networks.model import Policy  # type: ignore
        from rl.networks.storage import RolloutStorage  # type: ignore

        algo_args = default_algo_args(seed=0, num_env_steps=10_000, device_cuda=use_cuda)
        device = torch.device("cuda:0" if algo_args.cuda and torch.cuda.is_available() else "cpu")
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

        def one_update(acc: Dict[str, float] | None = None) -> None:
            t_act = t_env = t_boot = t_upd = 0.0
            for step in range(algo_args.num_steps):
                _sync(device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    rollouts_obs = {k: rollouts.obs[k][step] for k in rollouts.obs}
                    rollouts_hidden = {
                        k: rollouts.recurrent_hidden_states[k][step]
                        for k in rollouts.recurrent_hidden_states
                    }
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts_obs, rollouts_hidden, rollouts.masks[step]
                    )
                _sync(device)
                t_act += time.perf_counter() - t0

                if config.env.use_wrapper:
                    out_pred = rollouts_obs["spatial_edges"][:, :, 2:].to("cpu").numpy()
                    envs.talk2Env(out_pred)

                _sync(device)
                t1 = time.perf_counter()
                obs, reward, done, infos = envs.step(action)
                _sync(device)
                t_env += time.perf_counter() - t1
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

            _sync(device)
            t2 = time.perf_counter()
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
            _sync(device)
            t_boot += time.perf_counter() - t2

            _sync(device)
            t3 = time.perf_counter()
            agent.update(rollouts)
            _sync(device)
            t_upd += time.perf_counter() - t3
            rollouts.after_update()

            if acc is not None:
                acc["act"] += t_act
                acc["env"] += t_env
                acc["boot"] += t_boot
                acc["upd"] += t_upd

        for _ in range(warmup):
            one_update(None)

        acc = {"act": 0.0, "env": 0.0, "boot": 0.0, "upd": 0.0}
        _sync(device)
        t_all0 = time.perf_counter()
        for _ in range(timed):
            one_update(acc)
        _sync(device)
        dt = time.perf_counter() - t_all0
        steps = timed * algo_args.num_steps * algo_args.num_processes
        total_parts = sum(acc.values()) or 1.0
        envs.close()
        return {
            "mode": "hybrid(cuda policy+GST, CPU ORCA)" if device.type == "cuda" else "all-CPU",
            "device": str(device),
            "torch_threads": torch_threads,
            "env_steps": steps,
            "seconds": round(dt, 3),
            "steps_per_sec": round(steps / dt, 2),
            "pct_act": round(100.0 * acc["act"] / total_parts, 1),
            "pct_env": round(100.0 * acc["env"] / total_parts, 1),
            "pct_boot": round(100.0 * acc["boot"] / total_parts, 1),
            "pct_ppo_update": round(100.0 * acc["upd"] / total_parts, 1),
            "sec_act": round(acc["act"], 3),
            "sec_env": round(acc["env"], 3),
            "sec_boot": round(acc["boot"], 3),
            "sec_ppo_update": round(acc["upd"], 3),
        }
    finally:
        os.chdir(prev)


def main() -> None:
    rows: List[Dict[str, Any]] = []
    configs = [(False, 4)]
    if torch.cuda.is_available():
        configs = [(False, 4), (True, 4), (True, 2)]
    for use_cuda, th in configs:
        label = f"{'cuda' if use_cuda else 'cpu'} threads={th}"
        print(f"[bench] {label} ...", flush=True)
        try:
            row = _profile(use_cuda=use_cuda, torch_threads=th)
            rows.append(row)
            print(
                f"  -> {row['steps_per_sec']} steps/s | "
                f"env={row['pct_env']}% act={row['pct_act']}% "
                f"upd={row['pct_ppo_update']}% boot={row['pct_boot']}%",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL: {exc}", flush=True)

    print("\n=== RESULTS ===")
    for r in rows:
        print(r)

    if rows:
        best = max(rows, key=lambda r: r["steps_per_sec"])
        budget = 5 * 250_000 + 2 * max(0, 1_200_000 - 250_000)  # current script default
        sps = float(best["steps_per_sec"])
        print("\n=== BEST for current CARD budget (~3.15M RL steps) ===")
        print(best)
        print(f"RL-only ETA ~ {budget / sps / 3600:.1f} h @ {sps} steps/s")


if __name__ == "__main__":
    main()
