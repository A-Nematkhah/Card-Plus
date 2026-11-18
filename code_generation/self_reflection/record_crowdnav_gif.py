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

# Persistent matplotlib canvas for fallback RGB frames (GST native render is flaky on Agg).
_FIG = None
_AX = None


def _unwrap_sim(env):
    cur = env
    seen = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "robot") and hasattr(cur, "humans"):
            return cur
        cur = getattr(cur, "env", None) or getattr(cur, "unwrapped", None)
        if cur is env:
            break
    return getattr(env, "unwrapped", env)


def _render_frame(env):
    """Matplotlib top-down frame from sim state (GST native render needs UI axis)."""
    global _FIG, _AX
    import matplotlib.pyplot as plt
    from matplotlib import patches

    sim = _unwrap_sim(env)
    robot = getattr(sim, "robot", None)
    humans = list(getattr(sim, "humans", []) or [])
    if robot is None:
        return None
    if _FIG is None or _AX is None:
        _FIG, _AX = plt.subplots(figsize=(5, 5))
    ax = _AX
    ax.clear()
    arena = 7.0
    try:
        arena = float(sim.config.sim.arena_size) + 1.0
    except Exception:  # noqa: BLE001
        pass
    ax.set_xlim(-arena, arena)
    ax.set_ylim(-arena, arena)
    ax.set_aspect("equal")
    ax.set_title("circle-crossing-v0 (H=20)")
    pos = robot.get_position()
    ax.add_patch(plt.Circle((pos[0], pos[1]), float(robot.radius), color="gold", zorder=3))
    ax.plot([robot.gx], [robot.gy], marker="*", color="red", markersize=12)
    for human in humans:
        ax.add_patch(plt.Circle((human.px, human.py), float(human.radius), color="blue", alpha=0.55))
        ax.add_patch(
            patches.FancyArrowPatch(
                (human.px, human.py),
                (human.px + human.vx * 0.3, human.py + human.vy * 0.3),
                color="navy",
                arrowstyle="->",
                mutation_scale=8,
            )
        )
    _FIG.canvas.draw()
    canvas = _FIG.canvas
    w, h = canvas.get_width_height()
    if hasattr(canvas, "buffer_rgba"):
        buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        return np.ascontiguousarray(buf[:, :, :3])
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)


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
                frame = _render_frame(env)
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
