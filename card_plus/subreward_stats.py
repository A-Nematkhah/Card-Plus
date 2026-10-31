"""Offline sub-reward statistics feedback (no LLM)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from card_plus.config import CardPlusConfig
from card_plus.progress import compute_progress_values, traj_success_flag
from card_plus.reward_utils import (
    extract_component_arrays,
    load_totals_and_components_from_stored,
    recompute_reward_on_traj,
)
from card_plus.tpe_v2 import spearman_rho


SEVERITY_ORDER = {
    "DOMINANT": 0,
    "HIGHER_IN_FAILURES": 1,
    "NEG_CORR": 2,
    "SPARSE": 3,
    "INERT": 4,
}


@dataclass
class SubRewardRow:
    name: str
    mean_succ: Optional[float]
    mean_fail: Optional[float]
    share: float
    rho: float
    nz_frac: float
    flags: List[str] = field(default_factory=list)


def _per_step_mean(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.size == 0:
        return 0.0
    return float(np.mean(a))


def compute_subreward_rows(
    trajs: Sequence[Dict[str, Any]],
    cfg: CardPlusConfig,
    task_name: Optional[str] = None,
    traj_components: Optional[Sequence[Dict[str, np.ndarray]]] = None,
) -> Tuple[List[SubRewardRow], int, int, int]:
    n = len(trajs)
    succ_idx = [i for i, t in enumerate(trajs) if traj_success_flag(t)]
    fail_idx = [i for i, t in enumerate(trajs) if not traj_success_flag(t)]
    n_succ, n_fail = len(succ_idx), len(fail_idx)

    if traj_components is None:
        _, traj_components = load_totals_and_components_from_stored(list(trajs))

    all_keys: List[str] = []
    seen = set()
    for comps in traj_components:
        for k in comps:
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    progress = compute_progress_values(trajs, task_name=task_name)

    # Absolute share denominators
    abs_sums: Dict[str, float] = {k: 0.0 for k in all_keys}
    grand = 0.0
    for comps in traj_components:
        for k, arr in comps.items():
            s = float(np.sum(np.abs(arr)))
            abs_sums[k] = abs_sums.get(k, 0.0) + s
            grand += s
    if grand < 1e-12:
        grand = 1.0

    rows: List[SubRewardRow] = []
    for k in all_keys:
        succ_vals: List[float] = []
        fail_vals: List[float] = []
        all_vals: List[float] = []
        traj_means: List[float] = []
        for i, comps in enumerate(traj_components):
            arr = comps.get(k, np.zeros(0, dtype=np.float64))
            all_vals.extend(np.asarray(arr, dtype=np.float64).ravel().tolist())
            traj_means.append(_per_step_mean(arr) if arr.size else 0.0)
            if i in succ_idx:
                succ_vals.extend(np.asarray(arr, dtype=np.float64).ravel().tolist())
            else:
                fail_vals.extend(np.asarray(arr, dtype=np.float64).ravel().tolist())

        mean_succ = _per_step_mean(np.asarray(succ_vals)) if succ_vals else None
        mean_fail = _per_step_mean(np.asarray(fail_vals)) if fail_vals else None
        share = abs_sums.get(k, 0.0) / grand
        rho = spearman_rho(np.asarray(traj_means, dtype=np.float64), progress)
        arr_all = np.asarray(all_vals, dtype=np.float64) if all_vals else np.zeros(0)
        nz = float(np.mean(np.abs(arr_all) > 1e-12)) if arr_all.size else 0.0
        std = float(np.std(arr_all)) if arr_all.size else 0.0

        flags: List[str] = []
        if share > cfg.dominant_share:
            flags.append("DOMINANT")
        if (
            mean_succ is not None
            and mean_fail is not None
            and mean_fail > mean_succ
        ):
            flags.append("HIGHER_IN_FAILURES")
        if rho < cfg.neg_corr_threshold:
            flags.append("NEG_CORR")
        if std < cfg.inert_std_eps or nz == 0.0:
            flags.append("INERT")
        if nz < cfg.sparse_nonzero_frac:
            flags.append("SPARSE")

        rows.append(
            SubRewardRow(
                name=k,
                mean_succ=mean_succ,
                mean_fail=mean_fail,
                share=share,
                rho=rho,
                nz_frac=nz,
                flags=flags,
            )
        )
    return rows, n, n_succ, n_fail


def _flag_explanation(flag: str, name: str) -> str:
    explanations = {
        "DOMINANT": f"{name} accounts for most of the absolute reward mass and may drown other shaping terms.",
        "HIGHER_IN_FAILURES": f"{name} has a higher per-step mean on failed trajectories than on successful ones.",
        "NEG_CORR": f"{name} correlates negatively with task progress across trajectories.",
        "INERT": f"{name} is nearly constant or always zero and provides little learning signal.",
        "SPARSE": f"{name} is non-zero on only a small fraction of steps (sparse / bonus-like).",
    }
    return explanations.get(flag, f"{name}: flagged as {flag}.")


def format_subreward_stats_feedback(
    trajs: Sequence[Dict[str, Any]],
    cfg: CardPlusConfig,
    task_name: Optional[str] = None,
    traj_components: Optional[Sequence[Dict[str, np.ndarray]]] = None,
    header_note: str = "",
    version_index: Optional[int] = None,
) -> str:
    rows, n, n_succ, n_fail = compute_subreward_rows(
        trajs, cfg, task_name=task_name, traj_components=traj_components
    )
    version_bit = f" for reward version {version_index}" if version_index is not None else ""
    header = (
        f"# Sub-reward Statistics{version_bit} (computed offline from {n} trajectories: "
        f"{n_succ} successful, {n_fail} failed)"
    )
    if header_note:
        header = header + f"\n{header_note}"

    table_lines = [
        "name | mean per-step succ / fail | share of total |reward| | corr. with progress | non-zero steps"
    ]
    for r in rows:
        succ_s = "n/a" if r.mean_succ is None else f"{r.mean_succ:+.3f}"
        fail_s = "n/a" if r.mean_fail is None else f"{r.mean_fail:+.3f}"
        table_lines.append(
            f"- {r.name} | {succ_s} / {fail_s} | {r.share:.0%} | {r.rho:+.2f} | {r.nz_frac:.0%}"
        )

    findings: List[Tuple[int, str]] = []
    for r in rows:
        for flag in r.flags:
            sev = SEVERITY_ORDER.get(flag, 99)
            findings.append((sev, f"- [{flag}] {r.name}: {_flag_explanation(flag, r.name)}"))

    findings.sort(key=lambda x: x[0])

    # Cap total lines: header + blank + table header + rows + Findings: + findings
    # Drop lowest-severity findings first (highest sev number)
    body_prefix = [header, ""] + table_lines + ["Findings:"]
    # max_feedback_lines caps the whole block
    max_lines = cfg.max_feedback_lines
    remaining = max_lines - len(body_prefix)
    if remaining < 0:
        # Truncate table first if needed
        keep = max(0, max_lines - 3)
        body_prefix = [header, ""] + table_lines[:keep]
        finding_lines: List[str] = []
    else:
        # Keep highest severity (lowest sev number); drop from the end of sorted list
        selected = findings[:remaining] if remaining > 0 else []
        # If over budget somehow, drop lowest severity first = from end
        while len(body_prefix) + len(selected) > max_lines and selected:
            selected.pop()  # pop lowest severity among selected (last in sort)
        finding_lines = [t[1] for t in selected]

    if not finding_lines and any(r.flags for r in rows):
        # All findings dropped by cap — still note that
        finding_lines = []
    if not finding_lines and not any(r.flags for r in rows):
        finding_lines = ["- (none)"]

    lines = body_prefix + finding_lines
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    return "\n".join(lines)


def append_subreward_stats(
    feedback_str: str,
    trajs: Sequence[Dict[str, Any]],
    cfg: CardPlusConfig,
    task_name: Optional[str] = None,
    traj_components: Optional[Sequence[Dict[str, np.ndarray]]] = None,
    rl_skipped: bool = False,
    version_index: Optional[int] = None,
) -> str:
    if not cfg.use_subreward_stats:
        return feedback_str
    note = (
        "(candidate reward re-scored on pooled stored trajectories; RL skipped)"
        if rl_skipped
        else ""
    )
    block = format_subreward_stats_feedback(
        trajs,
        cfg,
        task_name=task_name,
        traj_components=traj_components,
        header_note=note,
        version_index=version_index,
    )
    if not feedback_str:
        return block
    return feedback_str.rstrip() + "\n\n" + block
