"""Publish a curated CrowdNav CARD results bundle under ``results/crowdnav/``.

Full training logs stay under ``self_reflection/logs/...``. This export keeps the
winner model, reward code, eval JSON, and GIFs in a short, related layout.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


def repo_root_from_here() -> Path:
    # .../code_generation/self_reflection/crowdnav_results.py -> repo root
    return Path(__file__).resolve().parents[2]


def infer_stack_tag(log_path: Path) -> str:
    name = log_path.name.lower()
    parts = {p.lower() for p in log_path.parts}
    if name.startswith("ppo_gst_h20") or "ppo_gst_1" in parts or "ppo_gst" in parts:
        return "ppo_gst_h20"
    if "sac_1" in parts or name.startswith("sac") or "sac" in parts:
        return "sac_h10"
    return "crowdnav"


def results_run_dir(log_path: Path, *, repo_root: Optional[Path] = None, tag: Optional[str] = None) -> Path:
    root = repo_root or repo_root_from_here()
    stamp = log_path.name
    stack = tag or infer_stack_tag(log_path)
    return root / "results" / "crowdnav" / f"{stamp}__{stack}"


def _copy_if_exists(src: Path, dest: Path) -> bool:
    if not src.is_file():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def publish_crowdnav_results(
    log_path: Path | str,
    *,
    repo_root: Optional[Path] = None,
    tag: Optional[str] = None,
) -> Path:
    """Copy winner artifacts from a CARD log dir into ``results/crowdnav/<run>/``."""
    log_path = Path(log_path)
    out = results_run_dir(log_path, repo_root=repo_root, tag=tag)
    models = out / "models"
    eval_dir = out / "eval"
    gifs = out / "gifs"
    models.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    gifs.mkdir(parents=True, exist_ok=True)

    best = log_path / "card_plus_best"
    copied = {
        "models": [],
        "eval": [],
        "gifs": [],
    }

    for name in ("best_model.pt", "best_model.zip", "specific.py", "version.txt"):
        if _copy_if_exists(best / name, models / name):
            copied["models"].append(name)

    for name in ("final_eval.json", "finalist_meta.json"):
        if _copy_if_exists(best / name, eval_dir / name):
            copied["eval"].append(name)

    report_src = log_path / "finalists" / "report.json"
    if _copy_if_exists(report_src, eval_dir / "finalists_report.json"):
        copied["eval"].append("finalists_report.json")

    # Prefer GIFs already saved next to the winner; also accept older names.
    gif_sources = []
    if best.is_dir():
        gif_sources.extend(sorted(best.glob("winner_rollout_seed*.gif")))
        gif_sources.extend(sorted(best.glob("*.gif")))
    # de-dupe while preserving order
    seen = set()
    unique_gifs = []
    for g in gif_sources:
        key = g.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique_gifs.append(g)
    for g in unique_gifs:
        # winner_rollout_seed1001.gif -> seed_1001.gif
        dest_name = g.name
        if dest_name.startswith("winner_rollout_seed") and dest_name.endswith(".gif"):
            dest_name = "seed_" + dest_name[len("winner_rollout_seed") :]
        if _copy_if_exists(g, gifs / dest_name):
            copied["gifs"].append(dest_name)

    final_eval = _load_json(eval_dir / "final_eval.json")
    report = _load_json(eval_dir / "finalists_report.json")
    meta = {
        "run_id": out.name,
        "source_log_path": str(log_path.resolve()),
        "stack_tag": tag or infer_stack_tag(log_path),
        "best_iteration": None if report is None else report.get("best_iteration"),
        "held_out_success_rate": None if final_eval is None else final_eval.get("success_rate"),
        "held_out_n_episodes": None if final_eval is None else final_eval.get("n_episodes"),
        "held_out_eval_seed": None if final_eval is None else final_eval.get("eval_seed"),
        "copied": copied,
        "layout": {
            "models": "winner checkpoint + reward code",
            "eval": "held-out + finalists report",
            "gifs": "rollout visualizations",
        },
    }
    (out / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out / "SOURCE.txt").write_text(
        f"Full CARD log (resume / debug):\n{log_path.resolve()}\n",
        encoding="utf-8",
    )
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Publish curated CrowdNav results bundle")
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()
    out = publish_crowdnav_results(args.log_path, tag=args.tag)
    print(f"Published results -> {out}")


if __name__ == "__main__":
    main()
