"""Across-step successive-halving finalists for CrowdNav CARD.

After each CARD step trains the single gate-best reward for a short budget
(e.g. 200k), this script picks the top-K versions from the archive and
*continues* each from the short-train checkpoint up to a long budget
(e.g. 400k total). Then runs a held-out deterministic eval on the winner.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from card_plus.archive import RewardArchive, VersionRecord, extract_success_curve
from card_plus.config import add_card_plus_arguments
from card_plus.dataset.gate import _json_safe, add_dataset_gate_arguments
from code_generation.self_reflection.utils import Logger, print_with_color

LOW_SCORE_SPREAD = 0.05


def _peak_success(v: VersionRecord) -> float:
    curve = v.proxy_curve or v.success_curve
    if curve:
        return float(max(curve))
    return float(v.version_score or 0.0)


def select_top_versions(
    versions: List[VersionRecord],
    top_k: int,
    *,
    low_spread: float = LOW_SCORE_SPREAD,
) -> List[VersionRecord]:
    """Keep top-K RL-ran versions.

    If ``version_score`` spread is tiny (hard envs), rank by peak then AUC /
    env progress. Otherwise rank by ``max(score, peak)``, then peak, then progress.
    """
    scored = [v for v in versions if v.rl_ran and v.reward_code.strip()]
    if not scored:
        return []
    scores = [float(v.version_score or 0.0) for v in scored]
    spread = max(scores) - min(scores) if scores else 0.0
    low_var = spread < float(low_spread)

    def key(v: VersionRecord) -> Tuple[float, float, float, float, float, float, int]:
        score = float(v.version_score or 0.0)
        peak = _peak_success(v)
        auc = float(v.curve_auc or 0.0)
        mean_prog = float(v.mean_progress) if v.mean_progress is not None else float("-1e18")
        mean_ret = float(v.mean_return) if v.mean_return is not None else float("-1e18")
        if low_var:
            return (peak, auc, mean_prog, mean_ret, score, max(score, peak), -int(v.iteration))
        return (max(score, peak), peak, mean_prog, mean_ret, auc, score, -int(v.iteration))

    ranked = sorted(scored, key=key, reverse=True)
    k = max(1, int(top_k))
    return ranked[: min(k, len(ranked))]


def short_checkpoint_path(log_path: Path, iteration: int, seed: int) -> Optional[Path]:
    """CARD step ``iteration`` trains ``code_{iteration-1}``.

    PPO/GST checkpoints are ``.pt`` only. Legacy SAC ``.zip`` files are ignored
    so callers do not silently "resume" into a cold start.
    """
    code_step = int(iteration) - 1
    if code_step < 0:
        return None
    tb = log_path / f"code_{code_step}" / f"seed_{seed}" / "tensorboard_log"
    for rel in (
        Path("latest_model.pt"),
        Path("eval") / "best_model.pt",
    ):
        p = tb / rel
        if p.is_file():
            return p
    return None


def resolve_finalist_model_path(code_dir: Path, seed: int) -> Optional[Path]:
    """Prefer PPO ``.pt`` artifacts written under a finalist code dir."""
    tb = code_dir / f"seed_{seed}" / "tensorboard_log"
    for rel in (
        Path("eval") / "best_model.pt",
        Path("latest_model.pt"),
        # Legacy SAC leftovers (held-out PPO loader cannot use these).
        Path("eval") / "best_model.zip",
        Path("latest_model.zip"),
    ):
        p = tb / rel
        if p.is_file():
            return p
    return None


def held_out_eval(
    args,
    *,
    model_path: str,
    reward_code: str,
    eval_seed: int,
    n_episodes: int,
) -> dict:
    """Deterministic rollout eval; seed must differ from training seed."""
    from code_generation.self_reflection.crowdnav_pp_env import make_dual_view_env
    from code_generation.self_reflection.crowdnav_ppo_train import load_policy_adapter

    model, _vec = load_policy_adapter(
        model_path, task_name=str(getattr(args, "task_name", "circle-crossing-v0")), seed=int(eval_seed)
    )
    _vec.close()
    env = make_dual_view_env(
        str(getattr(args, "task_name", "circle-crossing-v0")),
        seed=int(eval_seed),
        phase="test",
        use_ground_truth_reward=False,
        reward_code=reward_code,
        max_episode_steps=int(getattr(args, "max_episode_steps", 200)),
    )
    successes: List[float] = []
    returns: List[float] = []
    lengths: List[int] = []
    try:
        for _ in range(int(n_episodes)):
            model.reset_hidden()
            obs = env.reset()
            done = False
            ep_ret = 0.0
            steps = 0
            success = 0.0
            while not done and steps < int(getattr(args, "max_episode_steps", 200)):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                ep_ret += float(reward)
                steps += 1
                success = float(info.get("success", info.get("is_success", 0.0)) or 0.0)
                if success >= 1.0:
                    done = True
            successes.append(success)
            returns.append(ep_ret)
            lengths.append(steps)
    finally:
        env.close()

    succ_arr = np.asarray(successes, dtype=float)
    return {
        "eval_seed": int(eval_seed),
        "n_episodes": int(n_episodes),
        "success_rate": float(succ_arr.mean()) if succ_arr.size else 0.0,
        "success_std": float(succ_arr.std()) if succ_arr.size else 0.0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "successes": successes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train top CARD versions as SH finalists")
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument(
        "--train_max_steps",
        type=int,
        default=550_000,
        help="Target total timesteps after short+continue (e.g. 550k)",
    )
    parser.add_argument(
        "--short_train_steps",
        type=int,
        default=150_000,
        help="Steps already done in the CARD short screen (continue = target - short)",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--final_eval_seed", type=int, default=1001)
    parser.add_argument("--final_eval_episodes", type=int, default=50)
    parser.add_argument("--task_name", type=str, default="circle-crossing-v0")
    parser.add_argument("--benchmark_name", type=str, default="crowdnav")
    parser.add_argument("--train_num", type=int, default=4)
    parser.add_argument("--eval_num", type=int, default=4)
    parser.add_argument("--vec_env", type=str, default="auto", choices=["auto", "dummy", "subproc"])
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--sub_reward_eval_mult", type=int, default=2)
    parser.add_argument("--eval_freq", type=int, default=5_000)
    parser.add_argument("--max_episode_steps", type=int, default=200)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--trajectory_num", type=int, default=20)
    parser.add_argument("--llm_model_name", type=str, default="grok-4.3")
    parser.add_argument("--exp_name", type=str, default="card")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--entity", type=str, default="")
    add_card_plus_arguments(parser)
    add_dataset_gate_arguments(parser)
    args = parser.parse_args()

    if int(args.final_eval_seed) == int(args.seed):
        raise SystemExit(
            f"final_eval_seed ({args.final_eval_seed}) must differ from train seed ({args.seed})"
        )

    log_path = Path(args.log_path)
    std_file = log_path / "finalists_continue.log"
    logger = Logger(filename=str(std_file))
    sys.stdout = logger
    sys.stderr = logger

    # Heavy imports only when actually continuing training (keeps unit tests light).
    from code_generation.self_reflection.crowdnav_exp_one_step import train_and_eval

    archive = RewardArchive(run_dir=str(log_path))
    finalists = select_top_versions(archive.versions, args.top_k)
    if not finalists:
        raise SystemExit(
            f"No RL-ran versions in {log_path / 'card_plus_archive.json'}; "
            "run CARD steps with -CardPlus first."
        )

    target = int(args.train_max_steps)
    short_done = int(args.short_train_steps)
    extra = max(0, target - short_done)
    print_with_color(
        f"SH finalists top_k={args.top_k} target={target} short={short_done} "
        f"continue=+{extra}: "
        + ", ".join(
            f"iter={v.iteration} score={v.version_score} peak={_peak_success(v):.2f}"
            for v in finalists
        ),
        "BLUE",
    )

    out_root = log_path / "finalists"
    out_root.mkdir(parents=True, exist_ok=True)
    results: List[dict] = []
    best_peak = -1.0
    best_final = -1.0
    best_iteration: Optional[int] = None
    best_model_path: Optional[str] = None
    best_reward_code: Optional[str] = None

    for v in finalists:
        code_dir = out_root / f"iter_{v.iteration}"
        code_dir.mkdir(parents=True, exist_ok=True)
        specific_path = code_dir / "specific.py"
        with specific_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(v.reward_code)

        ckpt = short_checkpoint_path(log_path, int(v.iteration), int(args.seed))
        if ckpt is None:
            # Same total budget as resume path (short+extra == target), but cold start.
            print_with_color(
                f"Finalist iter={v.iteration}: no short checkpoint; "
                f"training full target={target} from scratch "
                f"(same total steps as resume path, no warm start)",
                "YELLOW",
            )
            learning_result, _, _ = train_and_eval(
                args=args,
                seed=args.seed,
                step=int(v.iteration),
                use_ground_truth_reward=False,
                specific_code=v.reward_code,
                specific_code_path=str(specific_path),
                load_model=False,
                train_steps=target,
            )
            continue_curve = extract_success_curve(learning_result) or []
            resumed = False
        else:
            # Seed finalist TB dir with short PPO checkpoint (and eval history if present).
            from code_generation.self_reflection.crowdnav_ppo_train import (
                is_usable_torch_checkpoint,
                read_checkpoint_meta,
            )

            tb = code_dir / f"seed_{args.seed}" / "tensorboard_log"
            eval_dir = tb / "eval"
            eval_dir.mkdir(parents=True, exist_ok=True)
            dest_model = tb / "latest_model.pt"
            if ckpt.suffix.lower() != ".pt":
                raise SystemExit(
                    f"Finalist iter={v.iteration}: expected PPO .pt checkpoint, got {ckpt}"
                )
            # If a previous finalist continue was interrupted, keep those weights
            # (do NOT re-copy the short ckpt and wipe progress).
            dest_meta = (
                read_checkpoint_meta(str(dest_model))
                if dest_model.is_file() and is_usable_torch_checkpoint(str(dest_model))
                else {}
            )
            dest_done = int(dest_meta.get("steps", 0) or 0)
            dest_complete = bool(dest_meta.get("complete"))
            reuse_partial = (
                dest_model.is_file()
                and is_usable_torch_checkpoint(str(dest_model))
                and dest_done > 0
                and not dest_complete
            )
            if not reuse_partial:
                shutil.copy2(ckpt, dest_model)
                short_eval = (
                    log_path
                    / f"code_{int(v.iteration) - 1}"
                    / f"seed_{args.seed}"
                    / "tensorboard_log"
                    / "eval"
                    / "evaluations.npz"
                )
                if short_eval.is_file():
                    shutil.copy2(short_eval, eval_dir / "evaluations.npz")
                dest_done = 0
                dest_complete = False
                print_with_color(
                    f"Finalist iter={v.iteration}: resume {ckpt.name} +{extra} -> {code_dir}",
                    "BLUE",
                )
            else:
                print_with_color(
                    f"Finalist iter={v.iteration}: continue partial "
                    f"{dest_model.name} done={dest_done}/{extra} -> {code_dir}",
                    "BLUE",
                )
            if extra <= 0 or dest_complete or (reuse_partial and dest_done >= int(extra)):
                learning_result = {}
                if (eval_dir / "evaluations.npz").is_file():
                    raw = np.load(eval_dir / "evaluations.npz")
                    learning_result = {k: raw[k] for k in raw.files}
                continue_curve = extract_success_curve(learning_result) or list(v.success_curve)
                resumed = True
            else:
                # Pass full continue budget; train_and_eval subtracts meta.steps itself.
                learning_result, _, _ = train_and_eval(
                    args=args,
                    seed=args.seed,
                    step=int(v.iteration),
                    use_ground_truth_reward=False,
                    specific_code=v.reward_code,
                    specific_code_path=str(specific_path),
                    load_model=False,
                    resume_model_path=str(dest_model),
                    train_steps=int(extra),
                )
                continue_curve = extract_success_curve(learning_result) or []
                resumed = True

        short_curve = list(v.success_curve or [])
        # Prefer full curve from continued evaluations.npz (includes copied short evals).
        full_curve = continue_curve if continue_curve else short_curve
        final_s = float(full_curve[-1]) if full_curve else 0.0
        peak_s = float(max(full_curve)) if full_curve else 0.0
        resolved = resolve_finalist_model_path(code_dir, int(args.seed))
        if resolved is not None and resolved.suffix.lower() == ".zip":
            print_with_color(
                f"Finalist iter={v.iteration}: only legacy .zip model found ({resolved.name}); "
                "held-out PPO eval needs .pt",
                "YELLOW",
            )
            resolved = None
        model_for_eval = str(resolved) if resolved is not None else ""

        entry = {
            "iteration": v.iteration,
            "short_version_score": v.version_score,
            "short_peak": _peak_success(v),
            "resumed_from_short": resumed,
            "extra_steps": extra if resumed else target,
            "final_success": final_s,
            "peak_success": peak_s,
            "success_curve": full_curve,
            "code_dir": str(code_dir),
            "model_path": model_for_eval,
        }
        results.append(entry)
        print_with_color(
            f"Finalist iter={v.iteration}: final={final_s:.2f} peak={peak_s:.2f} "
            f"(resumed={resumed})",
            "GREEN",
        )
        if (peak_s, final_s) > (best_peak, best_final):
            best_peak, best_final, best_iteration = peak_s, final_s, v.iteration
            best_model_path = model_for_eval or None
            best_reward_code = v.reward_code
            best_dir = log_path / "card_plus_best"
            best_dir.mkdir(parents=True, exist_ok=True)
            with (best_dir / "specific.py").open("w", encoding="utf-8", newline="\n") as f:
                f.write(v.reward_code)
            (best_dir / "version.txt").write_text(
                f"finalist_iter_{v.iteration}", encoding="utf-8"
            )
            (best_dir / "finalist_meta.json").write_text(
                json.dumps(_json_safe(entry), indent=2), encoding="utf-8"
            )
            if best_model_path and Path(best_model_path).is_file():
                shutil.copy2(best_model_path, best_dir / "best_model.pt")

    final_eval = None
    if (
        best_model_path
        and best_reward_code
        and Path(best_model_path).is_file()
        and Path(best_model_path).suffix.lower() == ".pt"
    ):
        print_with_color(
            f"Held-out final eval: model={best_model_path} "
            f"seed={args.final_eval_seed} n={args.final_eval_episodes}",
            "BLUE",
        )
        final_eval = held_out_eval(
            args,
            model_path=best_model_path,
            reward_code=best_reward_code,
            eval_seed=int(args.final_eval_seed),
            n_episodes=int(args.final_eval_episodes),
        )
        print_with_color(
            f"Held-out success={final_eval['success_rate']:.2%} "
            f"(±{final_eval['success_std']:.2%}) over {final_eval['n_episodes']} eps",
            "GREEN",
        )
        best_dir = log_path / "card_plus_best"
        best_dir.mkdir(parents=True, exist_ok=True)
        (best_dir / "final_eval.json").write_text(
            json.dumps(_json_safe(final_eval), indent=2), encoding="utf-8"
        )

    report = {
        "top_k": args.top_k,
        "train_max_steps": target,
        "short_train_steps": short_done,
        "continue_steps": extra,
        "selected_iterations": [v.iteration for v in finalists],
        "results": results,
        "best_iteration": best_iteration,
        "held_out_eval": final_eval,
    }
    report_path = out_root / "report.json"
    report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    print_with_color(f"Wrote finalists report to {report_path}", "GREEN")

    try:
        from code_generation.self_reflection.crowdnav_results import publish_crowdnav_results

        published = publish_crowdnav_results(log_path)
        print_with_color(f"Published curated results -> {published}", "GREEN")
    except Exception as exc:  # noqa: BLE001
        print_with_color(f"WARNING: results publish failed: {exc}", "YELLOW")


if __name__ == "__main__":
    main()
