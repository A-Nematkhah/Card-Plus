#!/usr/bin/env python
"""Phase-2 Reward Zoo calibration (composite ↔ online success).

Examples::

  # Offline-only zoo scores + difficulty report (no Go without online labels)
  python scripts/run_phase2_calibration.py \\
    --dataset results/datasets/crowdnav_ds_h15 \\
    --split val \\
    --harvest-runs results/runs/crowdnav/SAC_H15_2026-11-21-10-50-39 \\
    --base-reward path/to/specific.py \\
    --out results/phase2_calibration.json

  # With a zoo JSON that already has online_success labels
  python scripts/run_phase2_calibration.py \\
    --dataset results/datasets/crowdnav_ds_h15 \\
    --zoo results/reward_zoo.json \\
    --out results/phase2_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from card_plus.dataset.calibration import (
    ZooEntryScore,
    calibrate_reward_zoo,
    difficulty_report_for_reward,
    score_reward_on_episodes,
)
from card_plus.dataset.rescore import load_dataset_episodes
from card_plus.dataset.reward_zoo import (
    ZooReward,
    build_zoo,
    save_zoo_json,
)
from card_plus.dataset.learned_judge import load_judge
from card_plus.logging_utils import append_cost_event


def _load_base(path: Optional[str]) -> List[Tuple[str, str]]:
    if not path:
        return []
    p = Path(path)
    code = p.read_text(encoding="utf-8")
    return [(p.stem, code)]


def main() -> int:
    p = argparse.ArgumentParser(description="CARD+ Phase-2 reward zoo calibration")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=("all", "train", "val"))
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--zoo", type=str, default="", help="Existing zoo JSON (optional)")
    p.add_argument("--base-reward", type=str, default="", help="specific.py to mutate")
    p.add_argument(
        "--harvest-runs",
        nargs="*",
        default=[],
        help="Run dirs with code_*/candidates",
    )
    p.add_argument("--out", type=str, default="results/phase2_calibration.json")
    p.add_argument("--save-zoo", type=str, default="", help="Write assembled zoo JSON")
    p.add_argument("--spearman-threshold", type=float, default=0.5)
    p.add_argument("--min-n", type=int, default=20)
    p.add_argument("--max-rewards", type=int, default=40)
    p.add_argument("--no-mutations", action="store_true")
    p.add_argument(
        "--learned-judge",
        type=str,
        default="",
        help="Optional G2 judge JSON (results/learned_judge_g2.json)",
    )
    args = p.parse_args()
    judge = load_judge(args.learned_judge) if args.learned_judge else None

    rewards: List[ZooReward] = build_zoo(
        base_codes=_load_base(args.base_reward or None),
        run_dirs=args.harvest_runs or None,
        existing_zoo=args.zoo or None,
        include_mutations=not args.no_mutations,
    )
    if not rewards:
        print("No rewards in zoo. Pass --zoo, --base-reward, and/or --harvest-runs.")
        return 2
    rewards = rewards[: int(args.max_rewards)]
    if args.save_zoo:
        save_zoo_json(rewards, args.save_zoo)
        print(f"Wrote zoo ({len(rewards)}) → {args.save_zoo}")

    episodes = load_dataset_episodes(
        args.dataset,
        split=args.split,
        split_seed=int(args.split_seed),
    )
    print(f"Scoring {len(rewards)} rewards on {len(episodes)} episodes (split={args.split})")

    entries: List[ZooEntryScore] = []
    difficulty = None
    for i, r in enumerate(rewards):
        try:
            composite, metrics, _report = score_reward_on_episodes(
                r.code, episodes, learned_judge=judge
            )
        except Exception as exc:
            print(f"  skip {r.reward_id}: {exc}")
            continue
        meta = dict(r.meta or {})
        proxy = meta.get("online_proxy")
        for key in ("mean_progress", "version_score", "mean_return", "peak_success"):
            if proxy is None and meta.get(key) is not None:
                proxy = meta.get(key)
                break
        if proxy is None:
            proxy = r.online_success
        entries.append(
            ZooEntryScore(
                reward_id=r.reward_id,
                composite=composite,
                metrics=metrics,
                online_success=r.online_success,
                online_proxy=None if proxy is None else float(proxy),
                label_name=str(meta.get("primary_name") or "online_success"),
                source=r.source,
            )
        )
        print(
            f"  [{i+1}/{len(rewards)}] {r.reward_id}: composite={composite:.4f} "
            f"online={r.online_success} proxy={proxy} ({meta.get('primary_name')})"
        )
        if difficulty is None:
            try:
                difficulty = difficulty_report_for_reward(r.code, episodes)
            except Exception:
                difficulty = {}

    report = calibrate_reward_zoo(
        entries,
        spearman_threshold=float(args.spearman_threshold),
        min_n=int(args.min_n),
    )
    report.difficulty_metrics = difficulty or {}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["dataset"] = args.dataset
    payload["split"] = args.split
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    append_cost_event(
        str(out.parent),
        stage="phase2_calibration",
        llm_calls=0,
        extra={
            "n_rewards": len(entries),
            "spearman": report.spearman,
            "kendall": report.kendall,
            "go": report.go,
        },
    )
    if judge is not None and report.n < 20:
        report.go = False
        report.notes.append(
            "G2 in-sample Spearman is not Phase-2 Go (N<20). "
            "Use LOO from fit_learned_judge / phase2_decision.json."
        )
        payload = report.to_dict()
        payload["dataset"] = args.dataset
        payload["split"] = args.split
        payload["learned_judge"] = args.learned_judge
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Spearman={report.spearman:.3f} Kendall={report.kendall:.3f} "
        f"N={report.n} Go={report.go}"
    )
    for note in report.notes:
        print(f"  note: {note}")
    print(f"Wrote {out}")
    return 0 if (report.go or report.n < int(args.min_n)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
