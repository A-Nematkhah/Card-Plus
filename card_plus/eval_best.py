#!/usr/bin/env python
"""Final evaluation of the archived best CARD+ reward.

Reads ``<log_path>/card_plus_best/specific.py`` (written when
``--use_archive_rollback`` is on).

Selection during the CARD loop uses ``--seed`` (runners default to 1).
This entry point uses a **different** seed (default 1001) so the reported
success rate is not the same noisy number that picked the winner.

    python -m card_plus.eval_best --log_path PATH --benchmark metaworld
    python -m card_plus.eval_best --log_path PATH --benchmark crowdnav --dry_run
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

SELECTION_SEED_DEFAULT = 1
EVAL_SEED_DEFAULT = 1001


def best_code_path(log_path: str) -> Path:
    return Path(log_path) / "card_plus_best" / "specific.py"


def load_best_code(log_path: str) -> str:
    path = best_code_path(log_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No archived best reward at {path}. Run CARD+ with --use_archive_rollback."
        )
    return path.read_text(encoding="utf-8")


def load_run_args(log_path: str) -> Optional[Any]:
    """Best-effort: last rl_query_code_*.pkl in the run dir."""
    files = sorted(Path(log_path).glob("rl_query_code_*.pkl"))
    if not files:
        return None
    with files[-1].open("rb") as f:
        payload = pickle.load(f)
    return payload.get("rl_args")


def prepare_final_eval(
    log_path: str,
    *,
    eval_seed: int = EVAL_SEED_DEFAULT,
    selection_seed: int = SELECTION_SEED_DEFAULT,
) -> Dict[str, Any]:
    if int(eval_seed) == int(selection_seed):
        raise ValueError(
            f"eval_seed ({eval_seed}) must differ from selection_seed ({selection_seed})."
        )
    code = load_best_code(log_path)
    version_file = Path(log_path) / "card_plus_best" / "version.txt"
    version = version_file.read_text(encoding="utf-8").strip() if version_file.exists() else None
    return {
        "code": code,
        "code_path": str(best_code_path(log_path)),
        "version": version,
        "eval_seed": int(eval_seed),
        "selection_seed": int(selection_seed),
        "log_path": log_path,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate CARD+ archived best reward")
    p.add_argument("--log_path", type=str, required=True)
    p.add_argument("--benchmark", type=str, default="metaworld", choices=["metaworld", "crowdnav"])
    p.add_argument("--eval_seed", type=int, default=EVAL_SEED_DEFAULT)
    p.add_argument("--selection_seed", type=int, default=SELECTION_SEED_DEFAULT)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    info = prepare_final_eval(
        args.log_path, eval_seed=args.eval_seed, selection_seed=args.selection_seed
    )
    print(
        f"CARD+ final eval: version {info['version']} from {info['code_path']}\n"
        f"selection_seed={info['selection_seed']} eval_seed={info['eval_seed']} "
        f"(must differ)"
    )
    if args.dry_run:
        print("dry_run: not launching train_and_eval.")
        print(json.dumps({k: v for k, v in info.items() if k != "code"}, indent=2))
        return

    run_args = load_run_args(args.log_path)
    if run_args is None:
        raise SystemExit(f"No rl_query_code_*.pkl in {args.log_path}; cannot reconstruct train args.")
    run_args.seed = int(args.eval_seed)
    code_path = info["code_path"]
    cwd = _ROOT / "code_generation" / "self_reflection"
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(cwd))
    if args.benchmark == "metaworld":
        from metaworld_exp_one_step import train_and_eval
    else:
        from crowdnav_exp_one_step import train_and_eval
    print("Launching train_and_eval on archived best code with eval_seed "
          f"{args.eval_seed} (selection used seed {args.selection_seed}).")
    learning_result, _, _ = train_and_eval(
        args=run_args,
        seed=int(args.eval_seed),
        step=int(getattr(run_args, "step", 0) or 0),
        use_ground_truth_reward=False,
        specific_code=info["code"],
        specific_code_path=code_path,
        load_model=False,
    )
    out = Path(args.log_path) / "card_plus_best" / "final_eval.json"
    summary = {
        "eval_seed": args.eval_seed,
        "selection_seed": args.selection_seed,
        "version": info["version"],
        "keys": list(learning_result.keys()) if isinstance(learning_result, dict) else [],
    }
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
