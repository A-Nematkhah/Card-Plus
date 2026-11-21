#!/usr/bin/env python
"""Dry-run CARD+ loop (2 iterations) with mocked LLM and synthetic trajectories.

No RL training and no real LLM API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure repo root on path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from card_plus.archive import RewardArchive, VersionRecord, score_from_curve, success_curve_auc
from card_plus.config import CardPlusConfig
from card_plus.hooks import enrich_feedback, update_archive_after_version, write_iteration_log
from card_plus.logging_utils import append_cost_event
from card_plus.tpe_v2 import run_tpe_v2, should_skip_rl_for_tpe_v2


def _synthetic_pool(n: int = 12, hacked_sparse: bool = True) -> List[dict]:
    trajs: List[dict] = []
    for i in range(n):
        success = i < n // 2
        t = 16
        progress_end = 0.9 if success else 0.15
        xs = np.linspace(0.0, progress_end, t)
        obs = np.zeros((t, 39))
        obs[:, 4] = xs
        if hacked_sparse:
            dense = np.full(t, 0.01)
            sparse = np.zeros(t)
            if success:
                sparse[-1] = 10.0
        else:
            dense = xs * 2.0
            sparse = np.zeros(t)
        total = dense + sparse
        trajs.append(
            {
                "obs": obs,
                "actions": np.zeros((t, 4)),
                "rewards": total,
                "dones": np.zeros(t, dtype=bool),
                "infos": [{"success": success}] * t,
                "self.obj1.position": np.stack([xs, np.zeros(t), np.zeros(t)], axis=1),
                "self.goal_position": np.tile([1.0, 0.0, 0.0], (t, 1)),
                "sub_reward/approach": dense,
                "sub_reward/success_reward": sparse,
            }
        )
    return trajs


class MockLLM:
    """Deterministic Coder: returns two successive reward-code strings."""

    def __init__(self) -> None:
        self.calls = 0
        self.history: List[str] = []

    def generate(self, feedback: str, base_code: str) -> str:
        self.calls += 1
        self.history.append(feedback)
        if self.calls == 1:
            return (
                "def compute_dense_reward(self, action, obs):\n"
                "    return 1.0, {'approach': 0.01, 'success_reward': 10.0}\n"
            )
        return (
            "def compute_dense_reward(self, action, obs):\n"
            "    # improved dense shaping\n"
            "    return 2.0, {'approach': 1.0, 'success_reward': 0.5}\n"
        )


def run_dry_loop(
    run_dir: Path,
    cfg: CardPlusConfig,
    n_iters: int = 2,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    llm = MockLLM()
    archive = RewardArchive(run_dir=str(run_dir), cfg=cfg)
    code = "def compute_dense_reward(self, action, obs):\n    return 0.0, {}\n"
    feedback = ""
    summaries = []

    for it in range(1, n_iters + 1):
        code = llm.generate(feedback, code)
        pool = _synthetic_pool(hacked_sparse=(it == 1))
        # Fake v1: always pass for dry-run
        pass_v1, acc_v1 = True, 1.0
        tpe_v2 = None
        if cfg.use_tpe_v2:
            tpe_v2 = run_tpe_v2(
                pool,
                pass_v1=pass_v1,
                acc_v1=acc_v1,
                cfg=cfg,
                task_name="drawer-open-v2",
                use_stored_components=True,
            )
        skip = should_skip_rl_for_tpe_v2(cfg, tpe_v2) if tpe_v2 else False
        rl_ran = not skip
        # Synthetic learning curve
        if rl_ran:
            curve = [0.1 * it, 0.2 * it, 0.25 * it] if it == 1 else [0.5, 0.7, 0.8]
            learning_result = {"successes": np.asarray(curve), "results": np.asarray(curve)}
        else:
            curve = []
            learning_result = None

        feedback = "1. mock process feedback\n2. mock trajectory feedback"
        if skip:
            feedback = "preference-style feedback (RL skipped)"
        feedback = enrich_feedback(
            feedback,
            cfg,
            trajs=pool,
            task_name="drawer-open-v2",
            tpe_v2=tpe_v2,
            rl_skipped=skip,
        )
        record, rolled_back, notice, _restored = update_archive_after_version(
            archive,
            cfg,
            iteration=it,
            reward_code=code,
            sub_reward_names=["approach", "success_reward"],
            rl_ran=rl_ran,
            learning_result=learning_result,
            tpe_v2=tpe_v2,
            pass_v1=pass_v1,
            acc_v1=acc_v1,
            another_iteration_remains=(it < n_iters),
        )
        if rolled_back and notice:
            feedback = feedback.rstrip() + "\n\n" + notice
        write_iteration_log(
            str(run_dir),
            cfg,
            iteration=it,
            tpe_v2=tpe_v2,
            pass_v1=pass_v1,
            acc_v1=acc_v1,
            rl_ran=rl_ran,
            version_score=record.version_score,
            best_idx=archive.best_index(),
            rolled_back=rolled_back,
            token_counts={"prompt_tokens": 10, "completion_tokens": 20},
        )
        append_cost_event(
            str(run_dir),
            stage="dry_run_iteration",
            iteration=it,
            prompt_tokens=10,
            completion_tokens=20,
            llm_calls=1,
            rl_steps=0 if not rl_ran else 100,
            wall_s=0.0,
            extra={"rl_ran": rl_ran, "rolled_back": rolled_back},
        )
        summaries.append(
            {
                "iteration": it,
                "rl_ran": rl_ran,
                "hacked": None if tpe_v2 is None else tpe_v2.hacked,
                "rolled_back": rolled_back,
                "version_score": record.version_score,
                "best_idx": archive.best_index(),
            }
        )

    result = {
        "best_code": archive.best_code,
        "best_idx": archive.best_index(),
        "llm_calls": llm.calls,
        "iterations": summaries,
    }
    (run_dir / "dry_run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="CARD+ dry-run (mocked LLM, synthetic trajs)")
    p.add_argument("--run_dir", type=str, default="logs/card_plus_dry_run")
    p.add_argument("--use_archive_rollback", action="store_true")
    p.add_argument("--use_subreward_stats", action="store_true")
    p.add_argument("--use_tpe_v2", action="store_true")
    p.add_argument("--tpe_v2_mode", type=str, default="warn", choices=["warn", "gate"])
    p.add_argument("--all", action="store_true", help="Enable all CARD+ flags")
    args = p.parse_args()
    cfg = CardPlusConfig(
        use_archive_rollback=args.all or args.use_archive_rollback,
        use_subreward_stats=args.all or args.use_subreward_stats,
        use_tpe_v2=args.all or args.use_tpe_v2,
        tpe_v2_mode=args.tpe_v2_mode,
    )
    # Force any_enabled path for logging even if user forgot flags in quick test
    if not cfg.any_enabled:
        print("No CARD+ flags set; enabling --all for dry-run demonstration.")
        cfg = CardPlusConfig(
            use_archive_rollback=True,
            use_subreward_stats=True,
            use_tpe_v2=True,
            tpe_v2_mode=args.tpe_v2_mode,
        )
    run_dry_loop(Path(args.run_dir), cfg, n_iters=2)


if __name__ == "__main__":
    main()
