"""Roll out a trained CrowdNav SAC policy and save a GIF."""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import SAC

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from code_generation.self_reflection.crowdnav_env_utils import make_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
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

    print(f"Loading model: {args.model_path}")
    model = SAC.load(args.model_path)

    env_args = SimpleNamespace(
        task_name=args.task_name,
        max_episode_steps=args.max_steps,
        seed=args.seed,
    )

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
        env = make_env(
            env_args,
            seed=seed,
            max_episode_steps=args.max_steps,
            record_dir="logs/videos",
            is_eval=True,
            use_ground_truth_reward=True,
            reward_code=None,
        )()
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
                frame = env.render(mode="rgb_array")
                if frame is not None:
                    ep_frames.append(np.asarray(frame, dtype=np.uint8))
            step += 1
            if success >= 1.0:
                for _ in range(6):
                    frame = env.render(mode="rgb_array")
                    if frame is not None:
                        ep_frames.append(np.asarray(frame, dtype=np.uint8))
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

    if not frames:
        raise SystemExit("No frames captured; render(mode='rgb_array') returned nothing.")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    imageio.mimsave(out, frames, fps=args.fps, loop=0)
    print(f"saved {len(frames)} frames -> {out}")
    print(f"successes={successes} seeds={seeds_tried}")


if __name__ == "__main__":
    main()
