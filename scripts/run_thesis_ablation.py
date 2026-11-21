#!/usr/bin/env python
"""Orchestrate thesis ablation configs C0–C6 (Phase 6).

Writes a JSON plan of flag sets and optional shell commands. Does not launch
multi-day GPU jobs by itself — use the emitted commands / matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


ABLATIONS: Dict[str, Dict[str, Any]] = {
    "C0": {
        "desc": "CARD pure",
        "flags": {},
    },
    "C1": {
        "desc": "+ dataset gate",
        "flags": {
            "use_dataset_rescore": True,
            "dataset_mode": "gate",
        },
    },
    "C2": {
        "desc": "+ phase1 attribution/suicide",
        "flags": {
            "use_dataset_rescore": True,
            "dataset_gate_attribution": True,
            "dataset_suicide_probe": True,
        },
    },
    "C3": {
        "desc": "+ dataset split train",
        "flags": {
            "use_dataset_rescore": True,
            "dataset_split": "train",
        },
    },
    "C4": {
        "desc": "+ coeff optimize",
        "flags": {
            "use_dataset_rescore": True,
            "dataset_split": "train",
            "dataset_optimize_coeffs": True,
            "dataset_structure_prompt": True,
        },
    },
    "C5": {
        "desc": "+ asha screen / seed-aware rollback",
        "flags": {
            "use_dataset_rescore": True,
            "use_asha_screen": True,
            "asha_n_seeds": 2,
            "use_archive_rollback": True,
            "rollback_std_k": 1.0,
        },
    },
    "C6": {
        "desc": "CARD+ full",
        "flags": {
            "use_dataset_rescore": True,
            "dataset_gate_attribution": True,
            "dataset_suicide_probe": True,
            "dataset_split": "train",
            "dataset_optimize_coeffs": True,
            "use_asha_screen": True,
            "use_archive_rollback": True,
            "rollback_std_k": 1.0,
            "use_hypothesis_loop": True,
            "use_failure_cluster_feedback": True,
            "dataset_max_diff_gate": True,
        },
    },
}


def flags_to_cli(flags: Dict[str, Any]) -> List[str]:
    parts: List[str] = []
    for k, v in flags.items():
        if isinstance(v, bool):
            if v:
                parts.append(f"--{k}")
        else:
            parts.extend([f"--{k}", str(v)])
    return parts


def main() -> int:
    p = argparse.ArgumentParser(description="Emit thesis ablation matrix")
    p.add_argument("--out", default="results/thesis_ablation_matrix.json")
    p.add_argument("--codes", nargs="*", default=list(ABLATIONS.keys()))
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--dataset", default="results/datasets/crowdnav_ds_h15")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for code in args.codes:
        cfg = ABLATIONS[code]
        cli = flags_to_cli(cfg["flags"])
        if "use_dataset_rescore" in cfg["flags"]:
            cli.extend(["--dataset_path", args.dataset])
        rows.append(
            {
                "code": code,
                "desc": cfg["desc"],
                "flags": cfg["flags"],
                "cli": cli,
                "train_seeds": seeds,
                "eval_episodes": int(args.eval_episodes),
                "eval_seed": 1001,
            }
        )
    payload = {
        "protocol": {
            "n_train_seeds": len(seeds),
            "eval_episodes": int(args.eval_episodes),
            "eval_seed": 1001,
            "stats": "wilson_ci + paired_bootstrap via card_plus.stats_utils",
            "baselines": ["crowdnav_default", "CARD", "Text2Reward/Eureka-lite if available"],
        },
        "ablations": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} configs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
