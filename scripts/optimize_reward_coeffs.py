#!/usr/bin/env python
"""Phase-3 standalone coefficient optimization.

Example::

  python scripts/optimize_reward_coeffs.py \\
    --reward path/to/specific.py \\
    --dataset results/datasets/crowdnav_ds_h15 \\
    --method random --max-evals 24 \\
    --out results/coeff_optimize_demo.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from card_plus.dataset.coeff_optimize import (
    optimize_reward_coefficients,
    write_coeff_optimize_report,
)
from card_plus.dataset.rescore import load_dataset_episodes


def main() -> int:
    p = argparse.ArgumentParser(description="Optimize reward numeric coefficients")
    p.add_argument("--reward", required=True, help="specific.py path")
    p.add_argument("--dataset", required=True)
    p.add_argument("--method", default="auto", choices=("auto", "de", "cma", "random"))
    p.add_argument("--max-evals", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--suicide-probe", action="store_true")
    p.add_argument("--out", type=str, default="results/coeff_optimize.json")
    p.add_argument("--write-tuned", type=str, default="", help="Optional tuned .py path")
    args = p.parse_args()

    code = Path(args.reward).read_text(encoding="utf-8")
    train = load_dataset_episodes(args.dataset, split="train", split_seed=int(args.seed))
    val = load_dataset_episodes(args.dataset, split="val", split_seed=int(args.seed))
    result = optimize_reward_coefficients(
        code,
        train,
        val_episodes=val,
        method=str(args.method),
        max_evals=int(args.max_evals),
        seed=int(args.seed),
        use_suicide_probe=bool(args.suicide_probe),
    )
    write_coeff_optimize_report(result, args.out)
    if args.write_tuned:
        Path(args.write_tuned).write_text(result.tuned_code, encoding="utf-8")
        print(f"Wrote tuned reward → {args.write_tuned}")
    print(
        f"method={result.method} evals={result.n_evals} ok={result.ok}\n"
        f"train {result.train_composite_before:.4f} → {result.train_composite_after:.4f}\n"
        f"val   {result.val_composite_before} → {result.val_composite_after}"
    )
    for n in result.notes:
        print(f"  note: {n}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
