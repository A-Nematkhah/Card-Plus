"""Apply successive halfing to gate-accepted CrowdNav reward candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from card_plus.dataset.gate import _json_safe
from card_plus.dataset.successive_halving import (
    SHCandidate,
    parse_rung_steps,
    run_successive_halving,
    save_sh_report,
)
from code_generation.self_reflection.crowdnav_sh_train import train_candidate_rung
from code_generation.self_reflection.utils import print_with_color


def _load_candidates_from_disk(run_dir: Path, code_step: int) -> List[SHCandidate]:
    summary = run_dir / f"llm_candidates_{code_step}.json"
    cand_root = run_dir / f"code_{code_step}" / "candidates"
    if not summary.is_file() or not cand_root.is_dir():
        return []

    payload = json.loads(summary.read_text(encoding="utf-8"))
    out: List[SHCandidate] = []
    for item in payload.get("accepted", []):
        slot = int(item["slot"])
        slot_dir = cand_root / f"cand_{slot}"
        specific_path = slot_dir / "specific.py"
        general_path = slot_dir / "general.py"
        if not specific_path.is_file():
            print_with_color(f"SH: missing {specific_path}, skip slot={slot}", "YELLOW")
            continue
        general = general_path.read_text(encoding="utf-8") if general_path.is_file() else ""
        out.append(
            SHCandidate(
                slot=slot,
                attempt=int(item.get("attempt", 0)),
                specific_code=specific_path.read_text(encoding="utf-8"),
                general_code=general,
                composite=float(item.get("composite", 0.0)),
                code_path=str(specific_path),
            )
        )
    return out


def maybe_select_reward_via_sh(
    args: Any,
    *,
    run_dir: str,
    code_step: int,
    specific_code_path: str,
    general_code_path: str,
) -> Optional[str]:
    """If SH is enabled and enough candidates exist, overwrite the step reward.

    Returns the selected specific reward code, or None if SH did not run.
    """
    if not bool(getattr(args, "use_successive_halving", False)):
        return None

    run_path = Path(run_dir)
    candidates = _load_candidates_from_disk(run_path, code_step)
    min_c = int(getattr(args, "sh_min_candidates", 2))
    if len(candidates) < min_c:
        print_with_color(
            f"SH: {len(candidates)} candidate(s) on disk (<{min_c}); keep composite winner",
            "YELLOW",
        )
        return None

    rung_steps = parse_rung_steps(getattr(args, "sh_rung_steps", "20000,50000"))
    work_root = run_path / f"sh_step_{code_step}"

    def _train_fn(cand: SHCandidate, *, steps: int, resume_path, work_dir: str):
        return train_candidate_rung(
            args,
            reward_code=cand.specific_code,
            steps=steps,
            work_dir=work_dir,
            resume_path=resume_path,
            n_eval_episodes=int(getattr(args, "sh_eval_episodes", 20)),
        )

    result = run_successive_halving(
        candidates,
        train_fn=_train_fn,
        rung_steps=rung_steps,
        work_root=work_root,
        min_candidates=min_c,
    )
    save_sh_report(result, work_root / "sh_report.json")

    # Promote winner to the canonical CARD reward files for this code step.
    Path(specific_code_path).write_text(
        result.winner.specific_code, encoding="utf-8", newline="\n"
    )
    if result.winner.general_code:
        Path(general_code_path).write_text(
            result.winner.general_code, encoding="utf-8", newline="\n"
        )

    summary_path = run_path / f"llm_candidates_{code_step}.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["winner"] = {
            "slot": result.winner.slot,
            "attempt": result.winner.attempt,
            "composite": result.winner.composite,
            "success_rate": result.winner.success_rate,
            "selection": "successive_halving",
            "metrics": {"success_rate": result.winner.success_rate},
        }
        payload["sh_report"] = str(work_root / "sh_report.json")
        summary_path.write_text(
            json.dumps(_json_safe(payload), indent=2), encoding="utf-8"
        )

    print_with_color(
        f"SH promoted slot={result.winner.slot} -> {specific_code_path}",
        "GREEN",
    )
    return result.winner.specific_code
