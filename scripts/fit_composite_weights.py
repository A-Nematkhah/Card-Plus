#!/usr/bin/env python
"""Phase-2 Plan B: fit composite weights to maximize Spearman vs online_success.

Reads a labeled zoo + scored entries from phase2_calibration.json (or rescores).
Writes suggested weights for build_calibrated_metrics / NOTES.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from card_plus.dataset.scoring import calibrated_composite
from card_plus.tpe_v2 import spearman_rho


def _grid() -> List[Dict[str, float]]:
    out = []
    for wr in (0.05, 0.10, 0.20):
        for ws in (0.20, 0.30, 0.40):
            for wh in (0.20, 0.30, 0.40):
                wi = max(0.0, 1.0 - wr - ws - wh)
                if wi < 0.05:
                    continue
                out.append(dict(w_rank=wr, w_sep=ws, w_hack=wh, w_intra=wi))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--calibration", default="results/phase2_calibration.json")
    p.add_argument("--out", default="results/composite_weight_fit.json")
    args = p.parse_args()
    data = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    def _target(e: dict) -> Optional[float]:
        if e.get("online_proxy") is not None:
            return float(e["online_proxy"])
        if e.get("online_success") is not None:
            return float(e["online_success"])
        return None

    entries = [
        e for e in data.get("entries") or [] if _target(e) is not None and e.get("metrics")
    ]
    if len(entries) < 3:
        print(f"Need >=3 labeled entries, got {len(entries)}")
        return 2
    online = np.asarray([float(_target(e)) for e in entries])
    if float(np.std(online)) < 1e-8:
        print("Targets are constant — cannot fit weights. Use continuous proxies.")
        return 2
    best = None
    best_sp = -2.0
    for w in _grid():
        comps = []
        for e in entries:
            m = e["metrics"]
            comps.append(
                calibrated_composite(
                    offline_rank=float(m.get("offline_rank", float("nan"))),
                    separation=float(m.get("separation", float("nan"))),
                    anti_hack=float(m.get("anti_hack", float("nan"))),
                    intra_scenario=float(m.get("intra_scenario", float("nan"))),
                    **w,
                )
            )
        sp = float(spearman_rho(np.asarray(comps), online))
        if sp > best_sp:
            best_sp = sp
            best = dict(w)
            best["spearman"] = sp
    payload = {
        "n": len(entries),
        "best_weights": best,
        "baseline_spearman": data.get("spearman"),
        "note": "Apply best_weights in scoring.build_calibrated_metrics defaults after N>=20 confirm.",
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
