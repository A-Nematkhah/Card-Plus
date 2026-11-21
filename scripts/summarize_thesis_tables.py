#!/usr/bin/env python
"""Summarize thesis runs into cost / success tables with Wilson CI (Phase 6)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from card_plus.stats_utils import bootstrap_summary, format_ci, wilson_ci


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    archive = run_dir / "card_plus_archive.json"
    cost = _load_jsonl(run_dir / "card_plus_cost.jsonl")
    best_score = None
    if archive.is_file():
        data = json.loads(archive.read_text(encoding="utf-8"))
        scored = [
            float(v["version_score"])
            for v in data.get("versions") or []
            if v.get("version_score") is not None
        ]
        if scored:
            best_score = max(scored)
    prompt_tokens = sum(int(c.get("prompt_tokens") or 0) for c in cost)
    completion_tokens = sum(int(c.get("completion_tokens") or 0) for c in cost)
    llm_calls = sum(int(c.get("llm_calls") or 0) for c in cost)
    rl_steps = sum(int(c.get("rl_steps") or 0) for c in cost)
    wall_s = sum(float(c.get("wall_s") or 0.0) for c in cost)
    return {
        "run_dir": str(run_dir),
        "best_score": best_score,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "llm_calls": llm_calls,
        "rl_steps": rl_steps,
        "wall_s": wall_s,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run dirs or parent dirs; optionally prefix CODE=path",
    )
    p.add_argument("--out", default="results/thesis_summary.json")
    args = p.parse_args()

    by_code: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in args.runs:
        if "=" in item:
            code, path = item.split("=", 1)
            paths = [Path(path)]
        else:
            code = "UNK"
            root = Path(item)
            paths = [root] if (root / "card_plus_archive.json").is_file() else list(
                root.glob("*/")
            )
        for path in paths:
            if path.is_dir():
                by_code[code].append(summarize_run(path))

    table = []
    for code, rows in sorted(by_code.items()):
        successes = [r["best_score"] for r in rows if r["best_score"] is not None]
        # Treat scores as continuous; also report Wilson if they look Bernoulli.
        boot = bootstrap_summary([float(s) for s in successes], n_boot=1000, seed=0)
        bernoulli = successes and all(s in (0.0, 1.0) for s in successes)
        if bernoulli:
            w = wilson_ci(int(round(sum(successes))), len(successes))
            success_ci = format_ci(w.mean, w.low, w.high)
        else:
            success_ci = format_ci(boot["mean"], boot["mean_low"], boot["mean_high"])
        table.append(
            {
                "code": code,
                "n_runs": len(rows),
                "success_ci": success_ci,
                "mean_llm_calls": float(sum(r["llm_calls"] for r in rows) / max(len(rows), 1)),
                "mean_rl_steps": float(sum(r["rl_steps"] for r in rows) / max(len(rows), 1)),
                "mean_tokens": float(
                    sum(r["prompt_tokens"] + r["completion_tokens"] for r in rows)
                    / max(len(rows), 1)
                ),
                "runs": rows,
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"table": table}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
