#!/usr/bin/env python
"""Phase-5 E4: emit a fixed-pipeline coder-model ablation harness config.

Does not call APIs; writes a matrix of model ids for same gate/RL budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MODELS = [
    {"id": "primary", "desc": "current CARD Coder"},
    {"id": "reasoning", "desc": "stronger reasoning model"},
    {"id": "open_source", "desc": "open-weight coder"},
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/coder_model_ablation.json")
    p.add_argument("--token-budget", type=int, default=200000)
    p.add_argument("--rl-steps", type=int, default=200000)
    args = p.parse_args()
    payload = {
        "protocol": {
            "same_gate": True,
            "same_dataset": True,
            "token_budget": int(args.token_budget),
            "rl_steps": int(args.rl_steps),
            "n_seeds": 3,
        },
        "models": DEFAULT_MODELS,
        "metrics": ["final_success", "llm_calls", "tokens", "iterations_to_best"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
