#!/usr/bin/env python
"""Attach online labels to a reward zoo for Phase-2 Go/No-Go.

Modes:
  1) --from-archives: match zoo codes to archives (continuous signals).
  2) --import-archives: also **add** archive versions as zoo entries.
  3) --add-progress-baseline: inject a known progress-shaped control reward.
  4) --train: short SAC labeling (GPU); stores success + ep_rew proxy.

Primary calibration target (meta.online_proxy / primary_name):
  mean_progress → version_score → mean_return → non-zero peak/last success.
online_success keeps peak_success for reporting. Continuous fields always in meta.

Example::

  python scripts/label_reward_zoo_online.py \\
    --zoo results/reward_zoo.json \\
    --from-archives results/runs/crowdnav \\
    --import-archives --add-progress-baseline \\
    --out results/reward_zoo_labeled.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from card_plus.dataset.reward_zoo import ZooReward, load_zoo_json, save_zoo_json

PROGRESS_BASELINE_CODE = '''
def compute_dense_reward(self, action, obs):
    """Positive-control: potential-based progress + mild collision/success."""
    import numpy as np
    cur_d = float(np.linalg.norm(obs[0:2] - obs[4:6]))
    pre = getattr(self, "pre_obs", None)
    if pre is None:
        prev_d = cur_d
    else:
        prev_d = float(np.linalg.norm(pre[0:2] - pre[4:6]))
    r_progress = 1.0 * (prev_d - cur_d)
    comfort = 0.5
    if obs[7] < comfort:
        r_clearance = -5.0 * (comfort - float(obs[7]))
    else:
        r_clearance = 0.0
    total = r_progress + r_clearance
    collision = 0.0
    success = 0.0
    if float(obs[7]) < 0.0:
        collision = -100.0
        total += collision
        return total, {"progress": r_progress, "clearance": r_clearance, "collision": collision}
    if cur_d <= float(obs[6]):
        success = 500.0
        total += success
        return total, {"progress": r_progress, "clearance": r_clearance, "success": success}
    return total, {"progress": r_progress, "clearance": r_clearance}
'''


def _code_hash(code: str) -> str:
    return hashlib.sha1(code.strip().encode("utf-8")).hexdigest()[:16]


def _finite(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _curve_peak(curve: Any) -> Optional[float]:
    if not curve:
        return None
    try:
        arr = [float(x) for x in curve if x is not None]
    except (TypeError, ValueError):
        return None
    if not arr:
        return None
    return float(max(arr))


def _curve_last(curve: Any) -> Optional[float]:
    if not curve:
        return None
    try:
        return float(curve[-1])
    except (TypeError, ValueError, IndexError):
        return None


def primary_online_label(signals: Dict[str, Optional[float]]) -> Tuple[Optional[float], str]:
    """Pick calibration target: continuous proxies first, then non-zero success.

    All-zero peak_success collapses Spearman; mean_progress / version_score
    keep rank signal even when short SAC never succeeds.
    """
    peak = signals.get("peak_success")
    last = signals.get("last_success")
    vs = signals.get("version_score")
    prog = signals.get("mean_progress")
    ret = signals.get("mean_return")
    for name, val in (
        ("mean_progress", prog),
        ("version_score", vs),
        ("mean_return", ret),
    ):
        if val is not None:
            return float(val), name
    for name, val in (("peak_success", peak), ("last_success", last)):
        if val is not None and abs(float(val)) > 1e-8:
            return float(val), name
    if peak is not None:
        return float(peak), "peak_success"
    if last is not None:
        return float(last), "last_success"
    return None, "none"


def harvest_archive_entries(roots: List[Path]) -> List[Dict[str, Any]]:
    """All archive versions with code + continuous online signals."""
    rows: List[Dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("card_plus_archive.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for v in data.get("versions") or []:
                code = str(v.get("reward_code") or "")
                if not code.strip():
                    continue
                curve = v.get("success_curve") or v.get("proxy_curve") or []
                signals = {
                    "peak_success": _curve_peak(curve),
                    "last_success": _curve_last(curve),
                    "version_score": _finite(v.get("version_score")),
                    "mean_progress": _finite(v.get("mean_progress")),
                    "mean_return": _finite(v.get("mean_return")),
                    "online_success_field": _finite(v.get("online_success")),
                }
                if signals["online_success_field"] is not None:
                    # Explicit field wins as success-like.
                    signals["peak_success"] = signals["online_success_field"]
                primary, primary_name = primary_online_label(signals)
                if primary is None:
                    continue
                rows.append(
                    {
                        "code": code,
                        "code_hash": _code_hash(code),
                        "iteration": v.get("iteration"),
                        "source": str(path),
                        "primary": float(primary),
                        "primary_name": primary_name,
                        "signals": {k: signals[k] for k in signals if signals[k] is not None},
                    }
                )
    return rows


def harvest_archive_scores(roots: List[Path]) -> Dict[str, Dict[str, Any]]:
    """Map code_hash -> best (prefer higher primary) harvest row."""
    out: Dict[str, Dict[str, Any]] = {}
    for row in harvest_archive_entries(roots):
        h = row["code_hash"]
        prev = out.get(h)
        if prev is None or float(row["primary"]) >= float(prev["primary"]):
            out[h] = {
                "online_success": float(row["primary"]),
                "primary_name": row["primary_name"],
                "signals": dict(row["signals"]),
                "source": row["source"],
                "iteration": row["iteration"],
                "code": row["code"],
            }
    return out


def _apply_label(r: ZooReward, hit: Dict[str, Any]) -> None:
    signals = dict(hit.get("signals") or {})
    # Keep binary-ish success for reporting; continuous proxy for calibration.
    peak = signals.get("peak_success")
    r.online_success = float(peak) if peak is not None else float(hit["online_success"])
    r.meta = dict(r.meta or {})
    r.meta["label_source"] = hit.get("source")
    r.meta["code_hash"] = _code_hash(r.code)
    r.meta["primary_name"] = hit.get("primary_name")
    r.meta["online_proxy"] = float(hit["online_success"])
    for k, v in signals.items():
        r.meta[k] = v


def label_zoo_from_archives(
    rewards: List[ZooReward],
    archive_map: Dict[str, Dict[str, Any]],
) -> List[ZooReward]:
    labeled = 0
    for r in rewards:
        hit = archive_map.get(_code_hash(r.code))
        if hit is None:
            continue
        _apply_label(r, hit)
        labeled += 1
    print(f"Matched/labeled {labeled}/{len(rewards)} zoo rewards from archives")
    return rewards


def import_archive_versions(
    rewards: List[ZooReward],
    roots: List[Path],
    *,
    min_primary: float = 0.0,
    prefer_nonzero: bool = True,
) -> List[ZooReward]:
    """Append archive versions as zoo entries (for continuous calibration)."""
    by_id = {r.reward_id: r for r in rewards}
    by_hash = {_code_hash(r.code): r for r in rewards}
    added = 0
    rows = harvest_archive_entries(roots)
    if prefer_nonzero:
        rows = sorted(rows, key=lambda r: float(r["primary"]), reverse=True)
    for row in rows:
        if float(row["primary"]) < float(min_primary):
            continue
        h = row["code_hash"]
        rid = f"archive_iter{row.get('iteration')}_{h}"
        if h in by_hash:
            # Refresh label on existing
            _apply_label(by_hash[h], {
                "online_success": row["primary"],
                "primary_name": row["primary_name"],
                "signals": row["signals"],
                "source": row["source"],
            })
            continue
        if rid in by_id:
            continue
        peak = row["signals"].get("peak_success")
        zr = ZooReward(
            reward_id=rid,
            code=row["code"],
            source="archive_import",
            online_success=float(peak) if peak is not None else float(row["primary"]),
            meta={
                "label_source": row["source"],
                "code_hash": h,
                "primary_name": row["primary_name"],
                "online_proxy": float(row["primary"]),
                **row["signals"],
            },
        )
        rewards.append(zr)
        by_id[rid] = zr
        by_hash[h] = zr
        added += 1
    print(f"Imported {added} archive versions into zoo (total={len(rewards)})")
    return rewards


def ensure_progress_baseline(rewards: List[ZooReward]) -> List[ZooReward]:
    rid = "progress_baseline"
    if any(r.reward_id == rid for r in rewards):
        return rewards
    rewards.insert(
        0,
        ZooReward(
            reward_id=rid,
            code=PROGRESS_BASELINE_CODE.strip() + "\n",
            source="progress_baseline",
            meta={"role": "positive_control"},
        ),
    )
    print("Added progress_baseline positive-control reward")
    return rewards


def _make_train_args(seed: int, human_num: int) -> Any:
    from argparse import Namespace

    return Namespace(
        task_name="circle-crossing-v0",
        seed=int(seed),
        train_num=1,
        eval_num=1,
        max_episode_steps=100,
        sh_eval_episodes=10,
        human_num=int(human_num),
        device="cuda",
        torch_threads=1,
        eval_freq=5_000,
        benchmark_name="crowdnav",
        crowdnav_algo="sac",
    )


def _read_eval_aux(eval_dir: Path) -> Dict[str, float]:
    """Read continuous signals from EvalCallback npz if present."""
    path = eval_dir / "evaluations.npz"
    out: Dict[str, float] = {}
    if not path.is_file():
        return out
    data = np.load(path)
    if "results" in data.files:
        # shape (n_evals, n_episodes)
        res = np.asarray(data["results"], dtype=np.float64)
        if res.size:
            out["mean_return"] = float(np.nanmean(res[-1]))
    if "successes" in data.files:
        suc = np.asarray(data["successes"], dtype=np.float64)
        if suc.size:
            last = suc[-1]
            out["last_success"] = float(np.mean(last))
            out["peak_success"] = float(np.max(np.mean(suc, axis=-1)))
    return out


def train_label_reward(
    code: str,
    *,
    seeds: List[int],
    human_num: int,
    total_timesteps: int,
    run_dir: Path,
) -> Dict[str, float]:
    """Short SAC rung; return success + continuous proxies."""
    import os

    from code_generation.self_reflection.crowdnav_sh_train import train_candidate_rung

    os.environ["CARD_HUMAN_NUM"] = str(int(human_num))
    successes: List[float] = []
    returns: List[float] = []
    peaks: List[float] = []
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        args = _make_train_args(int(seed), int(human_num))
        success, _model = train_candidate_rung(
            args,
            reward_code=code,
            steps=int(total_timesteps),
            work_dir=str(seed_dir),
            seed=int(seed),
        )
        successes.append(float(success))
        aux = _read_eval_aux(seed_dir / "tensorboard_log" / "eval")
        if "mean_return" in aux:
            returns.append(float(aux["mean_return"]))
        if "peak_success" in aux:
            peaks.append(float(aux["peak_success"]))
    signals: Dict[str, float] = {
        "last_success": float(np.mean(successes)) if successes else 0.0,
        "peak_success": float(max(peaks)) if peaks else float(max(successes) if successes else 0.0),
    }
    if returns:
        signals["mean_return"] = float(np.mean(returns))
    primary, name = primary_online_label(
        {
            "peak_success": signals.get("peak_success"),
            "last_success": signals.get("last_success"),
            "mean_return": signals.get("mean_return"),
            "version_score": None,
            "mean_progress": None,
        }
    )
    signals["online_success"] = float(primary or 0.0)
    signals["primary_name"] = name  # type: ignore[assignment]
    return signals


def label_variance_report(rewards: List[ZooReward]) -> None:
    """Report variance on continuous calibration proxy, not just 0/1 success."""
    vals: List[float] = []
    for r in rewards:
        meta = dict(r.meta or {})
        proxy = meta.get("online_proxy")
        if proxy is None:
            for key in ("mean_progress", "version_score", "mean_return"):
                if meta.get(key) is not None:
                    proxy = meta[key]
                    break
        if proxy is None and r.online_success is not None:
            proxy = r.online_success
        if proxy is not None:
            vals.append(float(proxy))
    if not vals:
        print("Label variance: no labels")
        return
    arr = np.asarray(vals, dtype=np.float64)
    print(
        f"Label variance (proxy): n={arr.size} mean={arr.mean():.4f} "
        f"std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f} "
        f"unique≈{len(set(np.round(arr, 6)))}"
    )
    if float(arr.std()) < 1e-8:
        print(
            "WARNING: online proxies are constant — Spearman will be meaningless. "
            "Prefer --import-archives / continuous proxies, not all-zero short SAC."
        )


def main() -> int:
    p = argparse.ArgumentParser(description="Label reward zoo with online signals")
    p.add_argument("--zoo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--from-archives", nargs="*", default=[])
    p.add_argument(
        "--import-archives",
        action="store_true",
        help="Add archive versions as zoo entries (keeps continuous non-zero labels)",
    )
    p.add_argument(
        "--add-progress-baseline",
        action="store_true",
        help="Insert potential-based progress reward as positive control",
    )
    p.add_argument("--train", action="store_true", help="GPU short SAC labeling")
    p.add_argument("--seeds", type=str, default="0,1")
    p.add_argument("--human-num", type=int, default=10)
    p.add_argument("--timesteps", type=int, default=50_000)
    p.add_argument("--train-root", type=str, default="results/zoo_online_labels")
    p.add_argument("--max-rewards", type=int, default=80)
    p.add_argument(
        "--skip-labeled",
        action="store_true",
        default=True,
        help="Skip rewards that already have online_success when --train",
    )
    p.add_argument("--no-skip-labeled", action="store_false", dest="skip_labeled")
    args = p.parse_args()

    rewards = load_zoo_json(args.zoo)
    if args.add_progress_baseline:
        rewards = ensure_progress_baseline(rewards)
    rewards = rewards[: int(args.max_rewards)]

    archive_roots = [Path(r) for r in (args.from_archives or [])]
    if archive_roots:
        amap = harvest_archive_scores(archive_roots)
        rewards = label_zoo_from_archives(rewards, amap)
        if args.import_archives:
            rewards = import_archive_versions(rewards, archive_roots)

    if args.train:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
        root = Path(args.train_root)
        for i, r in enumerate(rewards):
            if args.skip_labeled and r.online_success is not None:
                continue
            print(f"[{i+1}/{len(rewards)}] training {r.reward_id} …")
            try:
                sig = train_label_reward(
                    r.code,
                    seeds=seeds,
                    human_num=int(args.human_num),
                    total_timesteps=int(args.timesteps),
                    run_dir=root / r.reward_id,
                )
                r.online_success = float(sig["online_success"])
                r.meta = dict(r.meta or {})
                r.meta["label_source"] = "sac_short"
                r.meta["seeds"] = seeds
                r.meta["primary_name"] = sig.get("primary_name")
                r.meta["online_proxy"] = float(sig["online_success"])
                for k in ("peak_success", "last_success", "mean_return"):
                    if k in sig:
                        r.meta[k] = float(sig[k])
                print(
                    f"  -> primary={r.online_success:.4f} "
                    f"({r.meta.get('primary_name')}) "
                    f"peak={r.meta.get('peak_success')} ret={r.meta.get('mean_return')}"
                )
            except Exception as exc:
                print(f"  failed: {exc}")

    n_lab = sum(1 for r in rewards if r.online_success is not None)
    label_variance_report(rewards)
    save_zoo_json(rewards, args.out)
    print(f"Wrote {args.out} ({n_lab}/{len(rewards)} labeled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
