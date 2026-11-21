"""Multi-candidate LLM reward selection via offline dataset gate.

Flow per CARD step (when dataset gate is on):
  1. Seed base prompts (caller).
  2. Fill ``n_candidates`` accepted rewards: rejected ones are discarded and
     rewritten with gate feedback until they pass (or retry budget ends).
  3. Among accepted candidates, keep the one with highest composite score.
  4. Persist only the winner as ``specific_{step}.py``; chat history keeps
     base prompts + winner response (rejects are not kept).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

from card_plus.archive import unified_diff_line_count
from card_plus.dataset.coeff_optimize import (
    optimize_reward_coefficients,
    write_coeff_optimize_report,
)
from card_plus.dataset.gate import (
    DatasetGateConfig,
    DatasetGateResult,
    _json_safe,
    run_dataset_gate,
)
from card_plus.dataset.rescore import load_dataset_episodes
from card_plus.dataset.reward_shaping import STRUCTURE_ONLY_PROMPT
from card_plus.logging_utils import append_cost_event
from code_generation.self_reflection.utils import print_with_color

# Soft online-learnability guidance (not a hard magnitude ban): gate can still
# pass huge collision terminals, but RL then sees ep_rew≈-700 and learns slowly.
_LEARNABILITY_HINT = (
    "Prefer a progress-dominated dense reward with mild terminals "
    "(collision/success on the same order as a few steps of progress, "
    "e.g. roughly tens not hundreds). Include an explicit success bonus. "
    "Avoid large per-step stall penalties that dominate every non-progress step. "
    "Outcome ranking must still be: success > timeout > collision."
)

def _alternative_prompt(structure_only: bool) -> str:
    base = (
        "Write a DIFFERENT reward function from any previous candidate. "
        "Use a distinct structure and weighting; do not lightly edit a prior version. "
    )
    if structure_only:
        base += STRUCTURE_ONLY_PROMPT + " "
    else:
        base += _LEARNABILITY_HINT + " "
    return base + "Return only one Python reward function in a ```python``` block."


ALTERNATIVE_PROMPT = _alternative_prompt(False)

_SUICIDE_RULE = (
    "Hard constraint: the collision terminal penalty must be strictly larger in "
    "magnitude than (episode_horizon * per_step_stall_or_time_penalty), so an "
    "early collision scores worse than waiting out a long no-progress episode."
)

def _reject_rewrite_template(*, suicide: bool, structure_only: bool) -> str:
    body = (
        "Your previous reward FAILED the offline dataset gate and is discarded.\n"
        "{gate_feedback}\n"
        "Rewrite a NEW reward with a different structure/weighting that ranks "
        "success trajectories above timeout and collision, encourages progress "
        "toward the goal, and does not reward fleeing or clearance farming. "
    )
    if structure_only:
        body += STRUCTURE_ONLY_PROMPT + " "
    else:
        body += _LEARNABILITY_HINT + " "
    if suicide:
        body += _SUICIDE_RULE + " "
    return body + "Return only one Python reward function in a ```python``` block."


REJECT_REWRITE_TEMPLATE = _reject_rewrite_template(suicide=False, structure_only=False)
REJECT_REWRITE_TEMPLATE_SUICIDE = _reject_rewrite_template(
    suicide=True, structure_only=False
)


@dataclass
class CandidateRecord:
    slot: int
    attempt: int
    general_code: str
    specific_code: str
    response: Any
    gate: Optional[DatasetGateResult]
    accepted: bool

    @property
    def composite(self) -> float:
        if self.gate is None:
            return 0.0
        return float(self.gate.composite)


@dataclass
class CandidateSelectionResult:
    winner: CandidateRecord
    accepted: List[CandidateRecord] = field(default_factory=list)
    all_attempts: List[CandidateRecord] = field(default_factory=list)


def learnability_bonus(gate: Optional[DatasetGateResult]) -> float:
    """Soft online-RL preference among gate passers (never rejects).

    Huge |mean_collided| / return_std pass the offline gate by separating
    outcomes, but poison SAC/PPO early learning. This bonus is small so a
    clearly better composite still wins; it mainly breaks near-ties toward
    progress-scale rewards with a positive success mean.
    """
    if gate is None:
        return 0.0
    m = gate.metrics or {}
    mean_s = float(m.get("mean_success", 0.0) or 0.0)
    mean_c = abs(float(m.get("mean_collided", 0.0) or 0.0))
    mean_t = abs(float(m.get("mean_timeout", 0.0) or 0.0))
    ret_std = abs(float(m.get("return_std", 0.0) or 0.0))
    bonus = 0.0
    if mean_s > 0.0:
        bonus += 0.04
    # log soft penalties: -20 → mild, -500/-1000 → several points of composite
    bonus -= 0.035 * math.log10(1.0 + mean_c / 25.0)
    bonus -= 0.020 * math.log10(1.0 + mean_t / 40.0)
    bonus -= 0.020 * math.log10(1.0 + ret_std / 40.0)
    return float(bonus)


def selection_score(rec: CandidateRecord) -> float:
    return float(rec.composite) + learnability_bonus(rec.gate)


def _maybe_optimize_coeffs(
    specific_code: str,
    *,
    gate_cfg: DatasetGateConfig,
    run_dir: Optional[str],
    step: int,
    suffix: str,
) -> Tuple[str, Optional[DatasetGateResult]]:
    """Optimize coeffs on train episodes; re-gate tuned code. Returns (code, gate)."""
    try:
        train_eps = load_dataset_episodes(
            gate_cfg.dataset_path,
            split="train",
            split_seed=int(gate_cfg.dataset_split_seed),
            val_fraction=float(gate_cfg.dataset_val_fraction),
        )
        val_eps = load_dataset_episodes(
            gate_cfg.dataset_path,
            split="val",
            split_seed=int(gate_cfg.dataset_split_seed),
            val_fraction=float(gate_cfg.dataset_val_fraction),
        )
    except Exception as exc:
        print_with_color(f"  coeff optimize skipped (load failed): {exc}", "YELLOW")
        gate = run_dataset_gate(
            gate_cfg,
            reward_code=specific_code,
            run_dir=run_dir,
            step=step,
            report_suffix=suffix,
        )
        return specific_code, gate

    result = optimize_reward_coefficients(
        specific_code,
        train_eps,
        val_episodes=val_eps,
        method=str(getattr(gate_cfg, "dataset_coeff_method", "auto")),
        max_evals=int(getattr(gate_cfg, "dataset_coeff_max_evals", 40)),
        seed=int(getattr(gate_cfg, "dataset_split_seed", 0)),
        use_suicide_probe=bool(getattr(gate_cfg, "dataset_suicide_probe", False)),
    )
    print_with_color(
        f"  coeff optimize ({result.method}): "
        f"train {result.train_composite_before:.4f}→{result.train_composite_after:.4f} "
        f"val {result.val_composite_before}→{result.val_composite_after} "
        f"evals={result.n_evals}",
        "BLUE",
    )
    if run_dir is not None:
        write_coeff_optimize_report(
            result,
            Path(run_dir) / f"coeff_optimize_{step}_{suffix}.json",
        )
        append_cost_event(
            run_dir,
            stage="coeff_optimize",
            iteration=step,
            extra={
                "method": result.method,
                "n_evals": result.n_evals,
                "train_before": result.train_composite_before,
                "train_after": result.train_composite_after,
                "val_before": result.val_composite_before,
                "val_after": result.val_composite_after,
                "ok": result.ok,
            },
        )
    tuned = result.tuned_code
    gate = run_dataset_gate(
        gate_cfg,
        reward_code=tuned,
        run_dir=run_dir,
        step=step,
        report_suffix=f"{suffix}_tuned",
    )
    return tuned, gate


def pick_best_accepted(accepted: List[CandidateRecord]) -> CandidateRecord:
    if not accepted:
        raise ValueError("pick_best_accepted requires at least one accepted candidate")
    return max(
        accepted,
        key=lambda r: (selection_score(r), -r.slot, -r.attempt),
    )


def select_best_reward_candidates(
    code_generator: Any,
    *,
    map_dict: dict,
    step: int,
    gate_cfg: DatasetGateConfig,
    n_candidates: int = 5,
    max_retries_per_slot: int = 5,
    run_dir: Optional[str] = None,
) -> CandidateSelectionResult:
    """Generate until ``n_candidates`` pass the gate, then pick max composite."""
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be >= 1, got {n_candidates}")
    if max_retries_per_slot < 1:
        raise ValueError(f"max_retries_per_slot must be >= 1, got {max_retries_per_slot}")
    if not gate_cfg.enabled:
        raise ValueError("select_best_reward_candidates requires an enabled dataset gate")

    base_history = list(code_generator.message_history)
    accepted: List[CandidateRecord] = []
    all_attempts: List[CandidateRecord] = []
    budget = int(n_candidates * max_retries_per_slot)
    llm_calls = 0
    slot = 0

    print_with_color(
        f"LLM candidates: target={n_candidates} retries/slot={max_retries_per_slot} "
        f"budget={budget}",
        "BLUE",
    )

    structure_only = bool(getattr(gate_cfg, "dataset_structure_prompt", False)) or bool(
        getattr(gate_cfg, "dataset_optimize_coeffs", False)
    )
    alt_prompt = _alternative_prompt(structure_only)
    rewrite_tmpl = _reject_rewrite_template(
        suicide=bool(getattr(gate_cfg, "dataset_suicide_probe", False)),
        structure_only=structure_only,
    )

    while len(accepted) < n_candidates and llm_calls < budget:
        code_generator.message_history = list(base_history)
        if slot > 0:
            code_generator.append_human(alt_prompt)

        slot_ok = False
        for attempt in range(max_retries_per_slot):
            if llm_calls >= budget:
                break
            llm_calls += 1
            try:
                response, general_code, specific_code = code_generator.propose_reward(
                    map_dict, step, generate_type=f"candidate_s{slot}_a{attempt}"
                )
            except RuntimeError as exc:
                print_with_color(
                    f"  candidate slot={slot} try={attempt}: codegen failed: {exc}",
                    "YELLOW",
                )
                continue
            suffix = f"cand{slot}_try{attempt}"
            gate = run_dataset_gate(
                gate_cfg,
                reward_code=specific_code,
                run_dir=run_dir,
                step=step,
                report_suffix=suffix,
            )
            keep = bool(gate is not None and gate.keep)

            # Phase-5: reject oversized diffs vs best on disk (when enabled).
            if keep and getattr(gate_cfg, "dataset_max_diff_gate", False) and run_dir:
                best_path = Path(run_dir) / "card_plus_best" / "specific.py"
                max_lines = int(getattr(gate_cfg, "max_diff_lines", 60) or 60)
                if best_path.is_file():
                    best_code = best_path.read_text(encoding="utf-8")
                    n_diff = unified_diff_line_count(best_code, specific_code)
                    if n_diff > max_lines:
                        keep = False
                        print_with_color(
                            f"  candidate slot={slot} try={attempt}: "
                            f"diff_gate reject lines={n_diff}>{max_lines}",
                            "YELLOW",
                        )

            # Phase-3: tune numeric coefficients on train split after a pass.
            if keep and getattr(gate_cfg, "dataset_optimize_coeffs", False):
                specific_code, gate = _maybe_optimize_coeffs(
                    specific_code,
                    gate_cfg=gate_cfg,
                    run_dir=run_dir,
                    step=step,
                    suffix=suffix,
                )

            rec = CandidateRecord(
                slot=slot,
                attempt=attempt,
                general_code=general_code,
                specific_code=specific_code,
                response=response,
                gate=gate,
                accepted=keep,
            )
            all_attempts.append(rec)
            print_with_color(
                f"  candidate slot={slot} try={attempt}: "
                f"keep={keep} composite={rec.composite:.4f}",
                "GREEN" if keep else "YELLOW",
            )

            if keep:
                accepted.append(rec)
                slot_ok = True
                break

            # Reject: keep branch only for rewrite; discarded from final history later.
            code_generator.message_history.append(response)
            feedback = gate.feedback_text() if gate is not None else "gate returned None"
            code_generator.append_human(
                rewrite_tmpl.format(gate_feedback=feedback)
            )

        if not slot_ok:
            mode = str(getattr(gate_cfg, "dataset_mode", "gate") or "gate").lower()
            # warn mode: after retries, soft-accept the best attempt of this slot.
            if mode == "warn":
                slot_attempts = [r for r in all_attempts if r.slot == slot]
                if slot_attempts:
                    best_slot = max(slot_attempts, key=lambda r: r.composite)
                    best_slot.accepted = True
                    accepted.append(best_slot)
                    slot_ok = True
                    print_with_color(
                        f"  slot={slot}: dataset_mode=warn soft-accept "
                        f"try={best_slot.attempt} composite={best_slot.composite:.4f}",
                        "YELLOW",
                    )
            if not slot_ok:
                print_with_color(
                    f"  slot={slot} exhausted retries without a passing reward",
                    "YELLOW",
                )
        slot += 1

    if not accepted:
        mode = str(getattr(gate_cfg, "dataset_mode", "gate") or "gate").lower()
        if mode == "warn" and all_attempts:
            best_fail = max(all_attempts, key=lambda r: r.composite)
            best_fail.accepted = True
            accepted = [best_fail]
            print_with_color(
                f"dataset_mode=warn: soft-accepting best failed candidate "
                f"slot={best_fail.slot} composite={best_fail.composite:.4f}",
                "YELLOW",
            )
        else:
            best_fail = (
                max(all_attempts, key=lambda r: r.composite) if all_attempts else None
            )
            detail = (
                f"best_fail_composite={best_fail.composite:.4f}"
                if best_fail is not None
                else "no attempts"
            )
            raise RuntimeError(
                f"No reward candidate passed the dataset gate after {llm_calls} LLM calls "
                f"({detail})."
            )

    if len(accepted) < n_candidates:
        print_with_color(
            f"Only {len(accepted)}/{n_candidates} candidates passed within budget; "
            f"selecting best among passers",
            "YELLOW",
        )

    winner = pick_best_accepted(accepted)
    win_learn = learnability_bonus(winner.gate)
    print_with_color(
        f"Selected among passers by composite+learnability: "
        f"composite={winner.composite:.4f} learnability={win_learn:+.4f} "
        f"score={selection_score(winner):.4f}",
        "BLUE",
    )
    code_generator.message_history = list(base_history) + [winner.response]
    code_generator.save_code(winner.general_code, winner.specific_code, step)

    # Persist every accepted candidate so successive halving can train them.
    if run_dir is not None:
        cand_root = Path(run_dir) / f"code_{step}" / "candidates"
        cand_root.mkdir(parents=True, exist_ok=True)
        for r in accepted:
            slot_dir = cand_root / f"cand_{r.slot}"
            slot_dir.mkdir(parents=True, exist_ok=True)
            # Path.write_text(..., newline=) needs Python 3.10+; card env is 3.8.
            with (slot_dir / "specific.py").open("w", encoding="utf-8", newline="\n") as f:
                f.write(r.specific_code)
            with (slot_dir / "general.py").open("w", encoding="utf-8", newline="\n") as f:
                f.write(r.general_code)
            meta = {
                "slot": r.slot,
                "attempt": r.attempt,
                "composite": r.composite,
                "metrics": r.gate.metrics if r.gate else {},
            }
            (slot_dir / "meta.json").write_text(
                json.dumps(_json_safe(meta), indent=2), encoding="utf-8"
            )

    # Canonical gate report for this step = winner.
    run_dataset_gate(
        gate_cfg,
        reward_code=winner.specific_code,
        run_dir=run_dir,
        step=step,
    )

    if run_dir is not None:
        out = Path(run_dir) / f"llm_candidates_{step}.json"
        retries = [r.attempt for r in all_attempts]
        median_retries = (
            float(sorted(retries)[len(retries) // 2]) if retries else 0.0
        )
        payload = {
            "step": step,
            "n_candidates_target": n_candidates,
            "n_accepted": len(accepted),
            "llm_calls": llm_calls,
            "llm_calls_per_accepted": (
                float(llm_calls) / float(len(accepted)) if accepted else None
            ),
            "median_retries": median_retries,
            "gate_pass_rate": (
                float(sum(1 for r in all_attempts if r.accepted)) / float(len(all_attempts))
                if all_attempts
                else None
            ),
            "winner": {
                "slot": winner.slot,
                "attempt": winner.attempt,
                "composite": winner.composite,
                "metrics": winner.gate.metrics if winner.gate else {},
                "selection": "composite+learnability",
            },
            "accepted": [
                {
                    "slot": r.slot,
                    "attempt": r.attempt,
                    "composite": r.composite,
                    "metrics": r.gate.metrics if r.gate else {},
                    "code_dir": str(Path(run_dir) / f"code_{step}" / "candidates" / f"cand_{r.slot}"),
                }
                for r in accepted
            ],
            "all_attempts": [
                {
                    "slot": r.slot,
                    "attempt": r.attempt,
                    "accepted": r.accepted,
                    "composite": r.composite,
                    "reasons": list(r.gate.reasons) if r.gate else [],
                }
                for r in all_attempts
            ],
        }
        out.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
        print_with_color(f"Wrote candidate summary to {out}", "GREEN")
        append_cost_event(
            run_dir,
            stage="llm_candidates",
            iteration=step,
            llm_calls=llm_calls,
            extra={
                "n_accepted": len(accepted),
                "n_attempts": len(all_attempts),
                "llm_calls_per_accepted": payload["llm_calls_per_accepted"],
                "median_retries": median_retries,
                "gate_pass_rate": payload["gate_pass_rate"],
                "winner_composite": winner.composite,
            },
        )

    print_with_color(
        f"Selected winner slot={winner.slot} try={winner.attempt} "
        f"composite={winner.composite:.4f} "
        f"(accepted={len(accepted)}/{n_candidates})",
        "GREEN",
    )
    return CandidateSelectionResult(
        winner=winner, accepted=accepted, all_attempts=all_attempts
    )
