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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from card_plus.dataset.gate import (
    DatasetGateConfig,
    DatasetGateResult,
    _json_safe,
    run_dataset_gate,
)
from code_generation.self_reflection.utils import print_with_color

ALTERNATIVE_PROMPT = (
    "Write a DIFFERENT reward function from any previous candidate. "
    "Use a distinct structure and weighting; do not lightly edit a prior version. "
    "Return only one Python reward function in a ```python``` block."
)

REJECT_REWRITE_TEMPLATE = (
    "Your previous reward FAILED the offline dataset gate and is discarded.\n"
    "{gate_feedback}\n"
    "Rewrite a NEW reward with a different structure/weighting that ranks "
    "success trajectories above timeout and collision, encourages progress "
    "toward the goal, and does not reward fleeing or clearance farming. "
    "Return only one Python reward function in a ```python``` block."
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


def pick_best_accepted(accepted: List[CandidateRecord]) -> CandidateRecord:
    if not accepted:
        raise ValueError("pick_best_accepted requires at least one accepted candidate")
    return max(accepted, key=lambda r: (r.composite, -r.slot, -r.attempt))


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

    while len(accepted) < n_candidates and llm_calls < budget:
        code_generator.message_history = list(base_history)
        if slot > 0:
            code_generator.append_human(ALTERNATIVE_PROMPT)

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
                REJECT_REWRITE_TEMPLATE.format(gate_feedback=feedback)
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
    code_generator.message_history = list(base_history) + [winner.response]
    code_generator.save_code(winner.general_code, winner.specific_code, step)

    # Persist every accepted candidate so successive halving can train them.
    if run_dir is not None:
        cand_root = Path(run_dir) / f"code_{step}" / "candidates"
        cand_root.mkdir(parents=True, exist_ok=True)
        for r in accepted:
            slot_dir = cand_root / f"cand_{r.slot}"
            slot_dir.mkdir(parents=True, exist_ok=True)
            (slot_dir / "specific.py").write_text(
                r.specific_code, encoding="utf-8", newline="\n"
            )
            (slot_dir / "general.py").write_text(
                r.general_code, encoding="utf-8", newline="\n"
            )
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
        payload = {
            "step": step,
            "n_candidates_target": n_candidates,
            "n_accepted": len(accepted),
            "llm_calls": llm_calls,
            "winner": {
                "slot": winner.slot,
                "attempt": winner.attempt,
                "composite": winner.composite,
                "metrics": winner.gate.metrics if winner.gate else {},
                "selection": "composite",
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

    print_with_color(
        f"Selected winner slot={winner.slot} try={winner.attempt} "
        f"composite={winner.composite:.4f} "
        f"(accepted={len(accepted)}/{n_candidates})",
        "GREEN",
    )
    return CandidateSelectionResult(
        winner=winner, accepted=accepted, all_attempts=all_attempts
    )
