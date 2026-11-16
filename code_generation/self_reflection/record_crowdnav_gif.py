"""Roll out a trained CrowdNav++ PPO/SRNN policy and save a GIF."""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import imageio.v2 as imageio
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from code_generation.self_reflection.crowdnav_pp_env import make_dual_view_env
from code_generation.self_reflection.crowdnav_ppo_train import load_policy_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to latest_model.pt")
    parser.add_argument("--task_name", type=str, default="circle-crossing-v0")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--every_n", type=int, default=1, help="Keep every Nth frame")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--prefer_success",
        action="store_true",
        help="Try extra seeds until one episode succeeds (cap: episodes*8)",
    )
    args = parser.parse_args()

    out = args.out
    if out is None:
        out = os.path.join(os.path.dirname(args.model_path), "crowdnav_winner_rollout.gif")

    print(f"Loading PPO/SRNN model: {args.model_path}")
    model, vec = load_policy_adapter(args.model_path, task_name=args.task_name, seed=args.seed)
    vec.close()

    frames = []
    successes = []
    seeds_tried = []
    seed = int(args.seed)
    max_tries = int(args.episodes) * (8 if args.prefer_success else 1)
    ep_done = 0
    tries = 0

    while ep_done < args.episodes and tries < max_tries:
        tries += 1
        seeds_tried.append(seed)
        env = make_dual_view_env(
            args.task_name,
            seed=seed,
            phase="test",
            use_ground_truth_reward=True,
            reward_code=None,
            max_episode_steps=args.max_steps,
        )
        model.reset_hidden()
        obs = env.reset()
        done = False
        step = 0
        success = 0.0
        ep_frames = []
        while not done and step < args.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            del reward
            success = float(info.get("success", info.get("is_success", 0.0)) or 0.0)
            if step % args.every_n == 0:
                # Dual-view wraps GST env; render via unwrapped if needed.
                render_env = env
                while hasattr(render_env, "env") and not hasattr(render_env, "render"):
                    render_env = render_env.env
                frame = None
                if hasattr(env, "env") and hasattr(env.env, "render"):
                    try:
                        frame = env.env.render("rgb_array")
                    except Exception:  # noqa: BLE001
                        frame = None
                if frame is not None:
                    ep_frames.append(np.asarray(frame, dtype=np.uint8))
            step += 1
            if success >= 1.0:
                done = True
        env.close()
        print(f"try seed={seed}: steps={step}, success={success}")
        if args.prefer_success and success < 1.0 and ep_done == 0:
            seed += 1
            continue
        frames.extend(ep_frames)
        successes.append(success)
        ep_done += 1
        seed += 1

    if frames:
        imageio.mimsave(out, frames, fps=args.fps)
        print(f"Wrote {out} ({len(frames)} frames), successes={successes}")
    else:
        print("No frames captured; GIF not written.")


if __name__ == "__main__":
    main()
