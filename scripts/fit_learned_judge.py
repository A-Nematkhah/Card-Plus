#!/usr/bin/env python
"""Fit G2 learned judge from phase2_calibration.json (continuous online_proxy)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from card_plus.dataset.learned_judge import (  # noqa: E402
    fit_learned_judge,
    redesigned_composite_weights,
    save_judge,
)
from card_plus.dataset.scoring import calibrated_composite  # noqa: E402
from card_plus.tpe_v2 import spearman_rho  # noqa: E402
import numpy as np


def _target(e: dict):
    if e.get("online_proxy") is not None:
        return float(e["online_proxy"])
    if e.get("online_success") is not None:
        return float(e["online_success"])
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--calibration", default="results/phase2_calibration.json")
    p.add_argument("--out", default="results/learned_judge_g2.json")
    p.add_argument("--decision", default="results/phase2_decision.json")
    args = p.parse_args()

    data = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    labeled = [e for e in (data.get("entries") or []) if _target(e) is not None and e.get("metrics")]
    if len(labeled) < 3:
        print(f"Need >=3 labeled entries, got {len(labeled)}")
        return 2

    metrics_list = [e["metrics"] for e in labeled]
    targets = [_target(e) for e in labeled]
    judge = fit_learned_judge(metrics_list, targets, target_name="online_proxy")
    save_judge(args.out, judge)

    # Baseline redesigned linear composite Spearman for comparison.
    w = redesigned_composite_weights()
    comps = [
        calibrated_composite(
            offline_rank=float(m.get("offline_rank", 0.0)),
            separation=float(m.get("separation", 0.0)),
            anti_hack=float(m.get("anti_hack", 0.0)),
            intra_scenario=float(m.get("intra_scenario", 0.0)),
            **w,
        )
        for m in metrics_list
    ]
    y = np.asarray(targets, dtype=np.float64)
    sp_v2 = float(spearman_rho(np.asarray(comps), y))
    preds = np.asarray([judge.score_metrics(m) for m in metrics_list])
    sp_g2 = float(spearman_rho(preds, y)) if float(np.std(preds)) > 1e-12 else 0.0

    decision = {
        "phase": 2,
        "decision": "No-Go",
        "reason": (
            "Composite (legacy) anti-correlated with continuous online_proxy "
            f"(Spearman={data.get('spearman')}); weight grid stayed negative; "
            "archive N too small for Go. Proceed to score redesign + G2."
        ),
        "n_labeled": len(labeled),
        "legacy_spearman": data.get("spearman"),
        "redesign_v2_weights": w,
        "redesign_v2_spearman": sp_v2,
        "g2_judge_path": str(args.out),
        "g2_spearman_train": judge.spearman_train,
        "g2_spearman_loo": judge.spearman_loo,
        "g2_spearman_refit_all": sp_g2,
        "go_threshold": 0.5,
        "phase3_unlocked": False,
        "next": "Use G2 judge behind flag; collect more continuous labels; re-evaluate Go.",
    }
    Path(args.decision).parent.mkdir(parents=True, exist_ok=True)
    Path(args.decision).write_text(json.dumps(decision, indent=2), encoding="utf-8")

    print(json.dumps({"judge": judge.to_dict(), "decision": decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
