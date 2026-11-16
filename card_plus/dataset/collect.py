"""Collect a CrowdNav smoke/full offline dataset.

Smoke::

    python -m card_plus.dataset.collect --out logs/crowdnav_ds_smoke --n-cases 16

Harder stratified set::

    python -m card_plus.dataset.collect --out logs/crowdnav_ds_v1 \\
        --pool 500 --n-cases 200 --min-near-path 2 \\
        --randomization with_random \\
        --policies go_to_goal,noisy_go_to_goal,stationary,random \\
        --random-seeds 0,1,2 --noise-std 0.3

With SAC + noisy SAC::

    python -m card_plus.dataset.collect --out logs/crowdnav_ds_v1 \\
        --pool 500 --n-cases 200 --min-near-path 2 \\
        --policies go_to_goal,stationary,random,sac,noisy_sac \\
        --sac-path path/to/latest_model.zip --noise-std 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from card_plus.dataset.labels import attach_labels
from card_plus.dataset.policies import get_policy
from card_plus.dataset.recorder import record_episode
from card_plus.dataset.scenarios import (
    DEFAULT_THIS_SEED,
    build_stratified_manifest,
    save_manifest,
    summarize_manifest,
)
from card_plus.dataset.schema import save_episode


def _parse_csv_strs(raw: str) -> List[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _expand_policy_jobs(
    policies: Sequence[str],
    *,
    random_seeds: Sequence[int],
    noise_seeds: Sequence[int],
    noise_std: float,
    sac_path: Optional[str],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Return list of (run_name, policy_key, kwargs)."""
    jobs: List[Tuple[str, str, Dict[str, Any]]] = []
    for pol in policies:
        key = pol.strip().lower()
        if key == "random":
            for seed in random_seeds:
                jobs.append((f"random_s{seed}", "random", {"seed": int(seed)}))
        elif key in ("noisy_go_to_goal", "noisy_goto"):
            for seed in noise_seeds:
                tag = f"noisy_goto_n{noise_std:g}_s{seed}"
                jobs.append(
                    (
                        tag,
                        "noisy_go_to_goal",
                        {"seed": int(seed), "noise_std": float(noise_std)},
                    )
                )
        elif key in ("sac", "sb3_sac"):
            if not sac_path:
                raise SystemExit("--sac-path is required when using policy 'sac'")
            jobs.append(("sac", "sac", {"model_path": sac_path}))
        elif key in ("noisy_sac", "sac_noisy"):
            if not sac_path:
                raise SystemExit("--sac-path is required when using policy 'noisy_sac'")
            for seed in noise_seeds:
                tag = f"noisy_sac_n{noise_std:g}_s{seed}"
                jobs.append(
                    (
                        tag,
                        "noisy_sac",
                        {
                            "model_path": sac_path,
                            "seed": int(seed),
                            "noise_std": float(noise_std),
                        },
                    )
                )
        else:
            jobs.append((key, key, {}))
    return jobs


def collect(
    out_dir: Path,
    *,
    n_cases: int = 16,
    pool: Optional[int] = None,
    case_offset: int = 0,
    this_seed: int = DEFAULT_THIS_SEED,
    policies: Sequence[str] = ("go_to_goal", "stationary", "random"),
    sac_path: Optional[str] = None,
    max_steps: Optional[int] = None,
    random_seeds: Sequence[int] = (0,),
    noise_seeds: Sequence[int] = (0,),
    noise_std: float = 0.3,
    randomization: str = "no_random",
    min_near_path: float = 0.0,
    max_clearance: Optional[float] = None,
    corridor: float = 1.0,
    allow_soft_fill: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    env_config: Dict[str, Any] = {"randomization": str(randomization)}
    pool_n = int(pool) if pool is not None else int(n_cases)

    print(
        f"[collect] scanning pool={pool_n} keep={n_cases} seed={this_seed} "
        f"offset={case_offset} randomization={randomization} "
        f"min_near_path={min_near_path} max_clearance={max_clearance}"
    )
    t0 = time.time()
    specs = build_stratified_manifest(
        int(n_cases),
        pool=pool_n,
        min_near_path=float(min_near_path),
        max_clearance=max_clearance,
        this_seed=int(this_seed),
        case_offset=int(case_offset),
        env_config=env_config,
        corridor=float(corridor),
        allow_soft_fill=bool(allow_soft_fill),
    )
    save_manifest(specs, out_dir / "manifest.json")
    summary = summarize_manifest(specs)
    summary["pool_scanned"] = pool_n
    summary["min_near_path"] = float(min_near_path)
    summary["max_clearance"] = max_clearance
    summary["randomization"] = str(randomization)
    (out_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[collect] manifest done in {time.time() - t0:.1f}s — {summary}")

    jobs = _expand_policy_jobs(
        policies,
        random_seeds=random_seeds,
        noise_seeds=noise_seeds,
        noise_std=float(noise_std),
        sac_path=sac_path,
    )

    index = []
    for run_name, policy_key, kwargs in jobs:
        policy = get_policy(policy_key, **kwargs)
        print(f"[collect] policy={run_name} over {len(specs)} cases")
        for spec in specs:
            ep = record_episode(
                spec,
                policy,
                policy_name=run_name,
                max_steps=max_steps,
                env_config=env_config,
            )
            attach_labels(ep)
            fname = f"{spec.scenario_id}__{run_name}.pkl"
            path = save_episode(ep, episodes_dir / fname)
            row = {
                "path": str(path.relative_to(out_dir)).replace("\\", "/"),
                "scenario_id": spec.scenario_id,
                "test_case": spec.test_case,
                "policy": run_name,
                "T": ep.T,
                "outcome": ep.labels.outcome if ep.labels else None,
                "progress": ep.labels.progress if ep.labels else None,
                "humans_near_path": spec.difficulty.get("humans_near_path"),
                "min_clearance_to_path": spec.difficulty.get("min_clearance_to_path"),
            }
            index.append(row)
            print(
                f"  case={spec.test_case:04d} policy={run_name:22s} "
                f"T={ep.T:3d} outcome={row['outcome']}"
            )

    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    meta = {
        "n_cases": int(n_cases),
        "pool": pool_n,
        "case_offset": int(case_offset),
        "this_seed": int(this_seed),
        "policies": list(policies),
        "policy_runs": [j[0] for j in jobs],
        "random_seeds": list(random_seeds),
        "noise_seeds": list(noise_seeds),
        "noise_std": float(noise_std),
        "randomization": str(randomization),
        "min_near_path": float(min_near_path),
        "max_clearance": max_clearance,
        "n_episodes": len(index),
        "seconds": round(time.time() - t0, 2),
    }
    (out_dir / "collect_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[collect] wrote {len(index)} episodes → {out_dir}")
    return meta


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Collect CrowdNav offline dataset")
    p.add_argument("--out", type=str, required=True, help="Output directory")
    p.add_argument(
        "--n-cases",
        type=int,
        default=16,
        help="How many scenarios to keep after filtering (default 16)",
    )
    p.add_argument(
        "--pool",
        type=int,
        default=None,
        help="How many test cases to scan before keeping n-cases (default: n-cases)",
    )
    p.add_argument("--case-offset", type=int, default=0)
    p.add_argument("--this-seed", type=int, default=DEFAULT_THIS_SEED)
    p.add_argument(
        "--randomization",
        type=str,
        default="no_random",
        choices=("no_random", "with_random"),
        help="CrowdNav human attribute / mid-episode goal switching",
    )
    p.add_argument(
        "--min-near-path",
        type=float,
        default=0.0,
        help="Keep only scenarios with humans_near_path >= this (strict unless --allow-soft-fill)",
    )
    p.add_argument(
        "--max-clearance",
        type=float,
        default=None,
        help="Optional: keep only min_clearance_to_path <= this (meters)",
    )
    p.add_argument(
        "--allow-soft-fill",
        action="store_true",
        default=False,
        help="If too few scenarios pass filters, pad with softer ones (legacy behavior)",
    )
    p.add_argument("--corridor", type=float, default=1.0, help="Path corridor for difficulty")
    p.add_argument(
        "--policies",
        type=str,
        default="go_to_goal,stationary,random",
        help="Comma-separated: go_to_goal,noisy_go_to_goal,stationary,random,sac,noisy_sac",
    )
    p.add_argument("--sac-path", type=str, default=None, help="SB3 SAC .zip")
    p.add_argument("--max-steps", type=int, default=None, help="Cap horizon (default: env max)")
    p.add_argument(
        "--random-seeds",
        type=str,
        default="0",
        help="Comma-separated seeds for policy=random (e.g. 0,1,2)",
    )
    p.add_argument(
        "--noise-seeds",
        type=str,
        default="0",
        help="Comma-separated seeds for noisy_go_to_goal / noisy_sac",
    )
    p.add_argument("--noise-std", type=float, default=0.3, help="Action noise std for noisy_*")
    # Back-compat alias
    p.add_argument("--random-seed", type=int, default=None, help=argparse.SUPPRESS)
    args = p.parse_args(argv)

    random_seeds = _parse_int_list(args.random_seeds)
    if args.random_seed is not None:
        random_seeds = [int(args.random_seed)]

    collect(
        Path(args.out),
        n_cases=int(args.n_cases),
        pool=None if args.pool is None else int(args.pool),
        case_offset=int(args.case_offset),
        this_seed=int(args.this_seed),
        policies=_parse_csv_strs(args.policies),
        sac_path=args.sac_path,
        max_steps=args.max_steps,
        random_seeds=random_seeds,
        noise_seeds=_parse_int_list(args.noise_seeds),
        noise_std=float(args.noise_std),
        randomization=str(args.randomization),
        min_near_path=float(args.min_near_path),
        max_clearance=None if args.max_clearance is None else float(args.max_clearance),
        corridor=float(args.corridor),
        allow_soft_fill=bool(args.allow_soft_fill),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
