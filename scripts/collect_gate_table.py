#!/usr/bin/env python
"""Collect dataset_gate_*.json (+ optional archives) into one summary table.

Usage:
  python scripts/collect_gate_table.py --roots results/runs/crowdnav --out results/gate_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def collect_gate_rows(roots: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("dataset_gate*.json")):
            data = _load_json(path)
            if not data:
                continue
            metrics = data.get("metrics") or {}
            reasons = data.get("reasons") or []
            rows.append(
                {
                    "path": str(path),
                    "run_dir": str(path.parent),
                    "keep": bool(data.get("keep", False)),
                    "skip_rl": bool(data.get("skip_rl", False)),
                    "mode": data.get("mode"),
                    "composite": data.get("composite"),
                    "offline_rank": metrics.get("offline_rank"),
                    "separation": metrics.get("separation"),
                    "anti_hack": metrics.get("anti_hack"),
                    "intra_scenario": metrics.get("intra_scenario"),
                    "success_collided": metrics.get("success/collided"),
                    "success_timeout": metrics.get("success/timeout"),
                    "timeout_collided": metrics.get("timeout/collided"),
                    "progress_auc": metrics.get("progress_auc"),
                    "n_reasons": len(reasons),
                    "reasons": "; ".join(str(r) for r in reasons),
                }
            )
    return rows


def collect_archive_rows(roots: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("card_plus_archive.json")):
            data = _load_json(path)
            if not data:
                continue
            for v in data.get("versions") or []:
                rows.append(
                    {
                        "path": str(path),
                        "iteration": v.get("iteration"),
                        "version_score": v.get("version_score"),
                        "online_success": v.get("online_success"),
                        "gate_composite": v.get("gate_composite"),
                        "rl_ran": v.get("rl_ran"),
                        "hacked": v.get("hacked"),
                    }
                )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Collect gate/archive rows into a table")
    p.add_argument(
        "--roots",
        nargs="+",
        default=["results/runs/crowdnav"],
        help="Root directories to scan",
    )
    p.add_argument("--out", type=str, default="results/gate_summary.json")
    p.add_argument("--csv", type=str, default="", help="Optional CSV path for gate rows")
    args = p.parse_args()
    roots = [Path(r) for r in args.roots]
    gate_rows = collect_gate_rows(roots)
    archive_rows = collect_archive_rows(roots)
    n_keep = sum(1 for r in gate_rows if r.get("keep"))
    summary = {
        "n_gate_reports": len(gate_rows),
        "n_keep": n_keep,
        "gate_pass_rate": (float(n_keep) / float(len(gate_rows)) if gate_rows else None),
        "n_archive_versions": len(archive_rows),
        "gate_rows": gate_rows,
        "archive_rows": archive_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"Wrote {out} (gate={len(gate_rows)} keep={n_keep} "
        f"archives={len(archive_rows)})"
    )
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if gate_rows:
            fieldnames = list(gate_rows[0].keys())
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(gate_rows)
            print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
