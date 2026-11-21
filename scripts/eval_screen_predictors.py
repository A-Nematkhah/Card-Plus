#!/usr/bin/env python
"""Evaluate which short-curve screen metric best predicts final success (Phase 4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from card_plus.dataset.screen_rank import all_screen_metrics, rank_predictors


def _rows_from_archives(roots: List[Path]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("card_plus_archive.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            versions = data.get("versions") or []
            finals = [
                float(v["version_score"])
                for v in versions
                if v.get("rl_ran") and v.get("version_score") is not None
            ]
            if not finals:
                continue
            final_best = max(finals)
            for v in versions:
                if not v.get("rl_ran"):
                    continue
                succ = v.get("success_curve") or []
                proxy = v.get("proxy_curve") or []
                if not succ and not proxy:
                    continue
                metrics = all_screen_metrics(succ, proxy or None)
                metrics["final_success"] = float(
                    v.get("online_success")
                    if v.get("online_success") is not None
                    else (v.get("version_score") or final_best)
                )
                metrics["run"] = str(path)
                rows.append(metrics)
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", default=["results/runs/crowdnav"])
    p.add_argument("--out", default="results/screen_predictor_eval.json")
    args = p.parse_args()
    rows = _rows_from_archives([Path(r) for r in args.roots])
    keys = [
        "success_at_1",
        "success_at_3",
        "curve_auc",
        "curve_slope_3",
        "proxy_at_1",
        "combined",
    ]
    ranking = rank_predictors(rows, keys)
    payload = {"n_rows": len(rows), "ranking": ranking, "baseline_metric": "success_at_1"}
    if ranking:
        best = ranking[0]
        base = next((r for r in ranking if r["metric"] == "success_at_1"), None)
        payload["go"] = bool(
            base is not None and best["spearman"] >= base["spearman"] - 1e-12
        )
        payload["best_metric"] = best["metric"]
        payload["best_spearman"] = best["spearman"]
        payload["baseline_spearman"] = None if base is None else base["spearman"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
