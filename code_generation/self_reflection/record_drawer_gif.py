"""Roll out a trained SAC policy on drawer-open-v2 and save a GIF."""
import argparse
import os
import sys

# Init GLFW before mujoco_py offscreen render (fixes GLEW Missing GL version on Windows).
os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", r"C:\Users\AmirHossein\.mujoco\mujoco210")
os.environ.setdefault("MUJOCO_PY_MJKEY_PATH", r"C:\Users\AmirHossein\.mujoco\mjkey.txt")

import imageio.v2 as imageio
import metaworld.envs.mujoco.env_dict as _env_dict
import numpy as np
from mujoco_py import GlfwContext
from rlkit.envs.wrappers import NormalizedBoxEnv
from stable_baselines3 import SAC

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    GlfwContext(offscreen=True)
except Exception as exc:
    print(f"GlfwContext warning: {exc}")


def make_raw_env(task_name: str, seed: int):
    env_cls = _env_dict.ALL_V2_ENVIRONMENTS[task_name]
    env = env_cls()
    env._freeze_rand_vec = False
    env._set_task_called = True
    env = NormalizedBoxEnv(env=env, seed=seed)
    env.seed(seed)
    return env


def get_base_env(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def capture_frame(env, camera_name: str, resolution):
    base = get_base_env(env)
    frame = base.render(offscreen=True, camera_name=camera_name, resolution=resolution)
    # Keep the raw OpenGL buffer. np.flipud / rot90 invert it on this Windows
    # GlfwContext path (floor ends up on top).
    return np.ascontiguousarray(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=(
            r"D:\Thesis\Implementation\card-v1.0\code_generation\self_reflection\logs\metaworld"
            r"\grok-4.3\t=0.7\zero-shot\SAC_1\drawer-open-v2\card\exclude=None"
            r"\2026-10-25-23-09-03\code_2\seed_1\tensorboard_log\eval\best_model.zip"
        ),
    )
    parser.add_argument("--task_name", default="drawer-open-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--camera", default="corner2")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--every_n", type=int, default=2, help="Keep every Nth frame for smaller GIF")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--out",
        default=None,
        help="Output GIF path (default: next to model as drawer_open_rollout.gif)",
    )
    args = parser.parse_args()

    out = args.out
    if out is None:
        out = os.path.join(os.path.dirname(args.model_path), "drawer_open_rollout.gif")

    print(f"Loading model: {args.model_path}")
    model = SAC.load(args.model_path)

    env = make_raw_env(args.task_name, args.seed)
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
                frames.append(capture_frame(env, args.camera, (args.width, args.height)))
            step += 1
            if success >= 1.0:
                # hold a few end frames
                for _ in range(8):
                    frames.append(capture_frame(env, args.camera, (args.width, args.height)))
                done = True
        successes.append(success)
        print(f"episode {ep+1}: steps={step}, success={success}")

    env.close()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    imageio.mimsave(out, frames, fps=args.fps, loop=0)
    print(f"saved {len(frames)} frames -> {out}")
    print(f"successes={successes}")


if __name__ == "__main__":
    main()
