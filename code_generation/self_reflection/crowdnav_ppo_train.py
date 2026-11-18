"""CrowdNav++ PPO + selfAttn SRNN training backend for CARD CrowdNav."""
from __future__ import annotations

import json
import os
from collections import deque
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from code_generation.self_reflection.crowdnav_env_utils import (
    CROWDNAV_ROOT,
    _ensure_crowdnav_path,
    load_reward_function,
)
from code_generation.self_reflection.crowdnav_pp_env import make_dual_view_env, make_pp_vec_env


def checkpoint_meta_path(ckpt_path: str) -> str:
    return f"{ckpt_path}.meta.json"


def atomic_torch_save(
    obj: Any,
    path: str,
    *,
    steps: int = 0,
    complete: bool = False,
) -> None:
    """Write ``.pt`` via temp+replace so kills mid-save cannot leave a corrupt file."""
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)
    meta = {
        "steps": int(steps),
        "complete": bool(complete),
        "path": path,
    }
    meta_tmp = checkpoint_meta_path(path) + ".tmp"
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(meta_tmp, checkpoint_meta_path(path))


def read_checkpoint_meta(ckpt_path: str) -> Dict[str, Any]:
    meta_path = checkpoint_meta_path(ckpt_path)
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def is_usable_torch_checkpoint(path: str) -> bool:
    """True if file exists and ``torch.load`` can read it (rejects truncated saves)."""
    if not path or not os.path.isfile(path):
        return False
    if os.path.getsize(path) < 64:
        return False
    try:
        torch.load(path, map_location="cpu", weights_only=True)
        return True
    except Exception:
        try:
            # Older PyTorch / non-weights_only state_dicts
            torch.load(path, map_location="cpu")
            return True
        except Exception:
            return False


def remove_corrupt_checkpoint(path: str) -> None:
    for p in (path, f"{path}.tmp", checkpoint_meta_path(path), checkpoint_meta_path(path) + ".tmp"):
        try:
            if os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass


def default_algo_args(
    *,
    seed: int = 1,
    num_env_steps: int = 150_000,
    num_steps: int = 30,
    device_cuda: bool = True,
) -> SimpleNamespace:
    cuda = bool(device_cuda and torch.cuda.is_available())
    return SimpleNamespace(
        seed=int(seed),
        cuda=cuda,
        no_cuda=not cuda,
        cuda_deterministic=False,
        num_processes=1,
        num_mini_batch=1,
        num_steps=int(num_steps),
        num_env_steps=int(num_env_steps),
        ppo_epoch=5,
        clip_param=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.0,
        lr=4e-5,
        eps=1e-5,
        max_grad_norm=0.5,
        gamma=0.99,
        use_gae=True,
        gae_lambda=0.95,
        use_proper_time_limits=False,
        use_linear_lr_decay=False,
        algo="ppo",
        save_interval=50,
        log_interval=10,
        recurrent_policy=True,
        human_node_rnn_size=128,
        human_human_edge_rnn_size=256,
        human_node_input_size=3,
        human_human_edge_input_size=2,
        human_node_output_size=256,
        human_node_embedding_size=64,
        human_human_edge_embedding_size=64,
        attention_size=64,
        seq_length=30,
        use_self_attn=True,
        use_hr_attn=True,
        env_name="CrowdSimPredRealGST-v0",
        sort_humans=True,
        aux_loss=False,
    )


class PolicyPredictAdapter:
    """SB3-like ``predict`` over CrowdNav Policy + GST vec env (for collect_trajectories)."""

    def __init__(self, actor_critic, device: torch.device, algo_args: SimpleNamespace):
        self.actor_critic = actor_critic
        self.device = device
        self.algo_args = algo_args
        self._rnn_hxs = None
        self._masks = None

    def _init_hidden(self, batch: int = 1):
        edge_num = None
        # Infer human_num from spatial edges if available later; default 20+1.
        human_num = 20
        edge_num = human_num + 1
        self._rnn_hxs = {
            "human_node_rnn": torch.zeros(
                batch, 1, self.algo_args.human_node_rnn_size, device=self.device
            ),
            "human_human_edge_rnn": torch.zeros(
                batch, edge_num, self.algo_args.human_human_edge_rnn_size, device=self.device
            ),
        }
        self._masks = torch.ones(batch, 1, device=self.device)

    def _obs_to_batch(self, obs) -> Dict[str, torch.Tensor]:
        if isinstance(obs, dict):
            out = {}
            for k, v in obs.items():
                t = v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))
                # Dual-view gym obs can be float64/int; SRNN linears expect float32 on policy device.
                if k != "visible_masks" and t.dtype != torch.bool:
                    t = t.float()
                if t.dim() >= 1 and t.shape[0] != 1:
                    # single-env dict without batch dim
                    t = t.unsqueeze(0)
                out[k] = t.to(self.device)
            # Fix common shapes: robot_node (1,7) -> (1,1,7) already from env; ensure batch.
            for k, t in list(out.items()):
                if t.dim() == 2 and k in ("robot_node", "temporal_edges"):
                    out[k] = t.unsqueeze(0)
                elif t.dim() == 1 and k == "detected_human_num":
                    out[k] = t.view(1, 1).float()
                elif t.dim() == 1 and k == "visible_masks":
                    out[k] = t.unsqueeze(0)
                elif t.dim() == 2 and k == "spatial_edges":
                    out[k] = t.unsqueeze(0)
            return out
        raise TypeError("PolicyPredictAdapter expects Dict observations")

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True):
        batch_obs = self._obs_to_batch(obs)
        if self._rnn_hxs is None:
            # Rebuild edge hidden to match spatial_edges human count.
            human_num = int(batch_obs["spatial_edges"].shape[1])
            self._rnn_hxs = {
                "human_node_rnn": torch.zeros(
                    1, 1, self.algo_args.human_node_rnn_size, device=self.device
                ),
                "human_human_edge_rnn": torch.zeros(
                    1,
                    human_num + 1,
                    self.algo_args.human_human_edge_rnn_size,
                    device=self.device,
                ),
            }
            self._masks = torch.ones(1, 1, device=self.device)
        value, action, log_prob, self._rnn_hxs = self.actor_critic.act(
            batch_obs, self._rnn_hxs, self._masks, deterministic=deterministic
        )
        del value, log_prob
        act = action.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        # Env action space is Box([-1, 1]^2); unclipped samples break DualView steps.
        act = np.clip(act.reshape(-1), -1.0, 1.0).astype(np.float32, copy=False)
        return act, None

    def reset_hidden(self):
        self._rnn_hxs = None
        self._masks = None


def _eval_episodes(actor_critic, envs, device, algo_args, n_episodes: int) -> Dict[str, np.ndarray]:
    from card_plus.progress import episode_proxy_score

    adapter = PolicyPredictAdapter(actor_critic, device, algo_args)
    returns = []
    lengths = []
    successes = []
    progresses = []
    collideds = []
    soft_nears = []
    proxy_scores = []
    for _ in range(int(n_episodes)):
        adapter.reset_hidden()
        obs = envs.reset()
        done = False
        ep_ret = 0.0
        ep_len = 0
        success = 0.0
        progress = 0.0
        collided = 0.0
        soft_near = 0.0
        proxy = 0.0
        while not done:
            action, _ = adapter.predict(obs, deterministic=True)
            # Vec env expects batched actions.
            if action.ndim == 1:
                act_in = action.reshape(1, -1)
            else:
                act_in = action
            obs, reward, dones, infos = envs.step(act_in)
            if torch.is_tensor(reward):
                ep_ret += float(reward.reshape(-1)[0].item())
            else:
                ep_ret += float(np.asarray(reward).reshape(-1)[0])
            ep_len += 1
            done = bool(np.asarray(dones).reshape(-1)[0])
            info0 = infos[0] if isinstance(infos, (list, tuple)) else infos
            if isinstance(info0, dict):
                success = float(info0.get("success", info0.get("is_success", 0.0)) or 0.0)
                collided = float(info0.get("collided", 0.0) or 0.0)
                progress = float(info0.get("goal_progress", progress) or 0.0)
                soft_near = float(info0.get("soft_near", soft_near) or 0.0)
                if "proxy_score" in info0:
                    proxy = float(info0.get("proxy_score") or 0.0)
                else:
                    proxy = episode_proxy_score(
                        success=success,
                        progress=progress,
                        collided=collided,
                        soft_near=soft_near,
                    )
        returns.append(ep_ret)
        lengths.append(ep_len)
        successes.append(success)
        progresses.append(progress)
        collideds.append(collided)
        soft_nears.append(soft_near)
        proxy_scores.append(proxy)
    return {
        "results": np.asarray(returns, dtype=np.float64).reshape(1, -1),
        "ep_lengths": np.asarray(lengths, dtype=np.float64).reshape(1, -1),
        "successes": np.asarray(successes, dtype=np.float64).reshape(1, -1),
        "goal_progress": np.asarray(progresses, dtype=np.float64).reshape(1, -1),
        "collided": np.asarray(collideds, dtype=np.float64).reshape(1, -1),
        "soft_near": np.asarray(soft_nears, dtype=np.float64).reshape(1, -1),
        "proxy_scores": np.asarray(proxy_scores, dtype=np.float64).reshape(1, -1),
    }


def train_ppo(
    *,
    task_name: str,
    seed: int,
    total_steps: int,
    use_ground_truth_reward: bool,
    reward_code: Optional[str],
    save_dir: str,
    resume_path: Optional[str] = None,
    eval_freq_steps: int = 0,
    n_eval_episodes: int = 10,
    final_eval_episodes: int = 100,
    max_episode_steps: Optional[int] = None,
) -> Tuple[PolicyPredictAdapter, Dict[str, Any], str]:
    """Train CrowdNav PPO; returns predict-adapter, learning_result, checkpoint path.

    Mid-train evals run only if ``eval_freq_steps > 0``. After training, a single
    end-of-short eval with ``final_eval_episodes`` (default 100) is always run.
    """
    _ensure_crowdnav_path()
    prev_cwd = os.getcwd()
    os.chdir(CROWDNAV_ROOT)
    try:
        from rl import ppo as ppo_mod  # type: ignore
        from rl.networks.model import Policy  # type: ignore
        from rl.networks.storage import RolloutStorage  # type: ignore

        if (not use_ground_truth_reward) and reward_code:
            load_reward_function(code=reward_code)

        pref = str(os.environ.get("CARD_DEVICE", "auto")).strip().lower()
        want_cuda = pref == "cuda"
        algo_args = default_algo_args(
            seed=seed, num_env_steps=total_steps, device_cuda=want_cuda
        )
        envs, config, device = make_pp_vec_env(
            task_name,
            seed=seed,
            use_ground_truth_reward=use_ground_truth_reward,
            reward_code=reward_code,
            max_episode_steps=max_episode_steps,
            restore_llm_reward=(not use_ground_truth_reward),
        )
        # Keep policy no_cuda flag aligned with actual vec device.
        algo_args.cuda = device.type == "cuda"
        algo_args.no_cuda = device.type != "cuda"

        actor_critic = Policy(
            envs.observation_space.spaces,
            envs.action_space,
            base_kwargs=algo_args,
            base=config.robot.policy,
        )
        if resume_path and os.path.isfile(resume_path):
            if not is_usable_torch_checkpoint(resume_path):
                print(
                    f"[warn] Ignoring corrupt resume checkpoint: {resume_path}",
                    flush=True,
                )
                remove_corrupt_checkpoint(resume_path)
            else:
                actor_critic.load_state_dict(torch.load(resume_path, map_location=device))
                meta = read_checkpoint_meta(resume_path)
                print(
                    f"[resume] loaded {resume_path} "
                    f"(meta_steps={meta.get('steps', '?')} complete={meta.get('complete', False)})",
                    flush=True,
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

        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, "latest_model.pt")
        eval_dir = os.path.join(save_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)

        timesteps_log = []
        results_log = []
        lengths_log = []
        successes_log = []
        progress_log = []
        collided_log = []
        soft_near_log = []
        proxy_log = []
        episode_rewards = deque(maxlen=100)

        num_updates = max(1, int(algo_args.num_env_steps) // algo_args.num_steps // algo_args.num_processes)
        do_mid_eval = int(eval_freq_steps) > 0
        eval_every = (
            max(1, int(eval_freq_steps) // max(1, algo_args.num_steps * algo_args.num_processes))
            if do_mid_eval
            else None
        )
        end_eps = int(final_eval_episodes) if final_eval_episodes is not None else 0

        for j in range(num_updates):
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

                # Send GST predictions to env for optional rendering / sync.
                if config.env.use_wrapper:
                    out_pred = rollouts_obs["spatial_edges"][:, :, 2:].to("cpu").numpy()
                    ack = envs.talk2Env(out_pred)
                    assert all(ack)

                obs, reward, done, infos = envs.step(action)
                for info in infos:
                    if isinstance(info, dict) and "episode" in info:
                        episode_rewards.append(info["episode"]["r"])

                masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if (isinstance(info, dict) and "bad_transition" in info) else [1.0] for info in infos]
                )
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

            total_num_steps = (j + 1) * algo_args.num_processes * algo_args.num_steps
            # Mid-train eval for SH success curves (skip duplicate on last update;
            # end-of-short final_eval runs below with more episodes).
            if do_mid_eval and eval_every is not None and (j % eval_every == 0) and (j < num_updates - 1):
                metrics = _eval_episodes(
                    actor_critic, envs, device, algo_args, n_eval_episodes
                )
                timesteps_log.append(total_num_steps)
                results_log.append(metrics["results"].reshape(-1))
                lengths_log.append(metrics["ep_lengths"].reshape(-1))
                successes_log.append(metrics["successes"].reshape(-1))
                progress_log.append(metrics["goal_progress"].reshape(-1))
                collided_log.append(metrics["collided"].reshape(-1))
                soft_near_log.append(metrics["soft_near"].reshape(-1))
                proxy_log.append(metrics["proxy_scores"].reshape(-1))
                atomic_torch_save(
                    actor_critic.state_dict(),
                    ckpt_path,
                    steps=int(total_num_steps),
                    complete=False,
                )
                print(
                    f"[mid-eval] steps={total_num_steps} episodes={n_eval_episodes} "
                    f"success_rate={float(np.mean(metrics['successes'])):.3f} "
                    f"proxy={float(np.mean(metrics['proxy_scores'])):.3f} "
                    f"progress={float(np.mean(metrics['goal_progress'])):.3f}",
                    flush=True,
                )

        total_num_steps = num_updates * algo_args.num_processes * algo_args.num_steps
        atomic_torch_save(
            actor_critic.state_dict(),
            ckpt_path,
            steps=int(total_num_steps),
            complete=False,
        )

        # End-of-short evaluation only (default 100 episodes).
        if end_eps > 0:
            metrics = _eval_episodes(actor_critic, envs, device, algo_args, end_eps)
            timesteps_log.append(total_num_steps)
            results_log.append(metrics["results"].reshape(-1))
            lengths_log.append(metrics["ep_lengths"].reshape(-1))
            successes_log.append(metrics["successes"].reshape(-1))
            progress_log.append(metrics["goal_progress"].reshape(-1))
            collided_log.append(metrics["collided"].reshape(-1))
            soft_near_log.append(metrics["soft_near"].reshape(-1))
            proxy_log.append(metrics["proxy_scores"].reshape(-1))
            sr = float(np.mean(metrics["successes"]))
            print(
                f"[final-eval] steps={total_num_steps} episodes={end_eps} "
                f"success_rate={sr:.3f} "
                f"proxy={float(np.mean(metrics['proxy_scores'])):.3f} "
                f"progress={float(np.mean(metrics['goal_progress'])):.3f}",
                flush=True,
            )

        atomic_torch_save(
            actor_critic.state_dict(),
            ckpt_path,
            steps=int(total_num_steps),
            complete=True,
        )

        # Pad ragged eval rows with NaN (mid-eval may use fewer episodes than final).
        def _stack(rows):
            if not rows:
                width = end_eps if end_eps > 0 else int(n_eval_episodes)
                return np.zeros((0, width), dtype=np.float64)
            width = max(len(r) for r in rows)
            out = np.full((len(rows), width), np.nan, dtype=np.float64)
            for i, r in enumerate(rows):
                out[i, : len(r)] = r
            return out

        learning_result = {
            "timesteps": np.asarray(timesteps_log, dtype=np.int64),
            "results": _stack(results_log),
            "ep_lengths": _stack(lengths_log),
            "successes": _stack(successes_log),
            "goal_progress": _stack(progress_log),
            "collided": _stack(collided_log),
            "soft_near": _stack(soft_near_log),
            "proxy_scores": _stack(proxy_log),
        }
        np.savez(os.path.join(eval_dir, "evaluations.npz"), **learning_result)
        # Also write a compact summary for the end-of-short eval.
        if end_eps > 0 and successes_log:
            summary = {
                "final_eval_episodes": end_eps,
                "success_rate": float(np.mean(successes_log[-1])),
                "mean_return": float(np.mean(results_log[-1])),
                "mean_length": float(np.mean(lengths_log[-1])),
                "mean_progress": float(np.nanmean(progress_log[-1])),
                "collision_rate": float(np.nanmean(collided_log[-1])),
                "soft_near_rate": float(np.nanmean(soft_near_log[-1])),
                "proxy_score": float(np.nanmean(proxy_log[-1])),
                "timesteps": int(timesteps_log[-1]),
            }
            with open(os.path.join(eval_dir, "final_eval.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        adapter = PolicyPredictAdapter(actor_critic, device, algo_args)
        return adapter, learning_result, ckpt_path
    finally:
        os.chdir(prev_cwd)


def load_policy_adapter(
    checkpoint_path: str,
    *,
    task_name: str = "circle-crossing-v0",
    seed: int = 0,
) -> Tuple[PolicyPredictAdapter, Any]:
    """Load a ``.pt`` policy and return (adapter, vec_env) for rollouts."""
    _ensure_crowdnav_path()
    prev_cwd = os.getcwd()
    os.chdir(CROWDNAV_ROOT)
    try:
        from rl.networks.model import Policy  # type: ignore

        envs, config, device = make_pp_vec_env(
            task_name, seed=seed, use_ground_truth_reward=True, reward_code=None
        )
        # Must match CARD_DEVICE / vec device: SRNN allocates edge RNN zeros via args.no_cuda.
        algo_args = default_algo_args(
            seed=seed, num_env_steps=1, device_cuda=(device.type == "cuda")
        )
        algo_args.cuda = device.type == "cuda"
        algo_args.no_cuda = device.type != "cuda"
        actor_critic = Policy(
            envs.observation_space.spaces,
            envs.action_space,
            base_kwargs=algo_args,
            base=config.robot.policy,
        )
        if not is_usable_torch_checkpoint(checkpoint_path):
            raise RuntimeError(
                f"Corrupt or unreadable PPO checkpoint: {checkpoint_path}. "
                "Delete it (or re-run without --load_model) to retrain."
            )
        actor_critic.load_state_dict(torch.load(checkpoint_path, map_location=device))
        actor_critic.to(device)
        actor_critic.eval()
        return PolicyPredictAdapter(actor_critic, device, algo_args), envs
    finally:
        os.chdir(prev_cwd)
