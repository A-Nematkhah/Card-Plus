"""JSONL logging for CARD+ iterations."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


def append_card_plus_log(
    run_dir: str,
    record: Dict[str, Any],
    filename: str = "card_plus_log.jsonl",
) -> Path:
    path = Path(run_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def append_cost_event(
    run_dir: str,
    *,
    stage: str,
    iteration: Optional[int] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    llm_calls: Optional[int] = None,
    rl_steps: Optional[int] = None,
    wall_s: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    filename: str = "card_plus_cost.jsonl",
) -> Path:
    """Append one cost/accounting event to the run cost ledger."""
    record: Dict[str, Any] = {
        "ts": time.time(),
        "stage": str(stage),
        "iteration": iteration,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "llm_calls": llm_calls,
        "rl_steps": rl_steps,
        "wall_s": wall_s,
        "extra": extra or {},
    }
    return append_card_plus_log(run_dir, record, filename=filename)


def make_iteration_log(
    *,
    iteration: int,
    flags: Dict[str, Any],
    tpe_acc_v1: Optional[float] = None,
    tpe_pass_v1: Optional[bool] = None,
    rho_full: Optional[float] = None,
    rho_dense: Optional[float] = None,
    sparse_keys: Optional[list] = None,
    hacked: Optional[bool] = None,
    verdict: Optional[str] = None,
    rl_ran: bool = False,
    version_score: Optional[float] = None,
    best_idx: Optional[int] = None,
    rolled_back: bool = False,
    token_counts: Optional[Dict[str, Any]] = None,
    extra_score_final_version: bool = False,
    extra_train_cost: Optional[str] = None,
    diff_lines_vs_best: Optional[int] = None,
    diff_lines_vs_prev: Optional[int] = None,
    no_success_pool: Optional[bool] = None,
    pass_v1_overridden: Optional[bool] = None,
    is_final_step: Optional[bool] = None,
) -> Dict[str, Any]:
    return {
        "iteration": iteration,
        "flags": flags,
        "tpe_v1_acc": tpe_acc_v1,
        "tpe_v1_pass": tpe_pass_v1,
        "rho_full": rho_full,
        "rho_dense": rho_dense,
        "sparse_keys": sparse_keys or [],
        "hacked": hacked,
        "verdict": verdict,
        "rl_ran": rl_ran,
        "version_score": version_score,
        "best_idx": best_idx,
        "rolled_back": rolled_back,
        "token_counts": token_counts,
        "extra_score_final_version": extra_score_final_version,
        "extra_train_cost": extra_train_cost,
        "diff_lines_vs_best": diff_lines_vs_best,
        "diff_lines_vs_prev": diff_lines_vs_prev,
        "no_success_pool": no_success_pool,
        "pass_v1_overridden": pass_v1_overridden,
        "is_final_step": is_final_step,
    }
