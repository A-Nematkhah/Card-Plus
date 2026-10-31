"""Roll out a trained SAC policy on CrowdNav++ and save a GIF."""
import argparse
import os
import sys

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import SAC

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from code_generation.self_reflection.crowdnav_env_utils import make_crowdnav_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--task_name", default="circle-crossing-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--every_n", type=int, default=1)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = args.out or os.path.join(os.path.dirname(args.model_path), "crowdnav_rollout.gif")
    print(f"Loading model: {args.model_path}")
    model = SAC.load(args.model_path)

    env = make_crowdnav_env(args.task_name)
    env.seed(args.seed)
    frames = []
    successes = []

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        step = 0
        success = 0.0
        while not done and step < args.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            success = float(info.get("success", 0.0))
            if step % args.every_n == 0:
                frames.append(np.ascontiguousarray(env.render(mode="rgb_array")))
            step += 1
            if success >= 1.0:
                for _ in range(4):
                    frames.append(np.ascontiguousarray(env.render(mode="rgb_array")))
                done = True
        successes.append(success)
        print(f"episode {ep + 1}: steps={step}, success={success}")

    env.close()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    imageio.mimsave(out, frames, fps=args.fps, loop=0)
    print(f"saved {len(frames)} frames -> {out}")
    print(f"successes={successes}")


if __name__ == "__main__":
    main()
