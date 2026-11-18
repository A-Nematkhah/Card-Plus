"""Small hooks for integrating CARD+ into the Meta-World one-step loop.

When all CardPlusConfig flags are False, callers should skip these helpers so
baseline control flow stays bit-identical. Helpers themselves also no-op when
the relevant flag is False.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from card_plus.archive import (
    RewardArchive,
    VersionRecord,
    extract_mean_progress,
    extract_mean_return,
    extract_proxy_curve,
    extract_success_curve,
    score_from_curve,
    success_curve_auc,
    unified_diff_line_count,
)
from card_plus.config import CardPlusConfig
from card_plus.logging_utils import append_card_plus_log, make_iteration_log
from card_plus.loop_control import another_coder_query_follows, is_final_train_step
from card_plus.reward_utils import recompute_reward_on_traj
from card_plus.subreward_stats import append_subreward_stats
from card_plus.tpe_v2 import TPEv2Result, run_tpe_v2, should_skip_rl_for_tpe_v2

logger = logging.getLogger(__name__)


def run_optional_tpe_v2(
    cfg: CardPlusConfig,
    trajs: Sequence[Dict[str, Any]],
    *,
    pass_v1: bool,
    acc_v1: float,
    task_name: str,
    reward_fn: Optional[Callable[..., Any]] = None,
    benchmark_name: str = "metaworld",
) -> Optional[TPEv2Result]:
    if not cfg.use_tpe_v2:
        return None
    return run_tpe_v2(
        trajs,
        pass_v1=pass_v1,
        acc_v1=acc_v1,
        cfg=cfg,
        task_name=task_name,
        reward_fn=reward_fn,
        use_stored_components=(reward_fn is None),
        benchmark_name=benchmark_name,
    )


def enrich_feedback(
    feedback_str: str,
    cfg: CardPlusConfig,
    *,
    trajs: Sequence[Dict[str, Any]],
    task_name: str,
    tpe_v2: Optional[TPEv2Result],
    rl_skipped: bool,
    reward_fn: Optional[Callable[..., Any]] = None,
    benchmark_name: str = "metaworld",
    version_index: Optional[int] = None,
    version_score: Optional[float] = None,
) -> str:
    """Append dense-shaping then sub-reward stats after process/traj/preference text.

    Order: caller provides process/trajectory (or preference) first; this adds
    dense-shaping (if hacked), then stats. Rollback notice is appended by the
    caller *after* this function.
    """
    out = feedback_str
    suppress_hacked = (
        version_score is not None
        and float(version_score) >= float(cfg.suppress_hacked_feedback_score)
    )
    if cfg.use_tpe_v2 and tpe_v2 is not None and tpe_v2.feedback_text:
        if tpe_v2.hacked and not suppress_hacked:
            out = (out.rstrip() + "\n\n" + tpe_v2.feedback_text) if out else tpe_v2.feedback_text

    if cfg.use_subreward_stats and trajs and len(trajs) >= max(1, int(cfg.tpe_v2_min_trajs)):
        comps = None
        if reward_fn is not None and rl_skipped:
            comps_list = []
            for t in trajs:
                _, c = recompute_reward_on_traj(
                    reward_fn, t, benchmark_name=benchmark_name
                )
                comps_list.append(c)
            comps = comps_list
        out = append_subreward_stats(
            out,
            trajs,
            cfg,
            task_name=task_name,
            traj_components=comps,
            rl_skipped=rl_skipped,
            version_index=version_index,
        )
    elif cfg.use_subreward_stats and (not trajs or len(trajs) < max(1, int(cfg.tpe_v2_min_trajs))):
        version_bit = (
            f" for reward version {version_index}" if version_index is not None else ""
        )
        note = (
            f"# Sub-reward Statistics{version_bit}\n"
            "(deferred: trajectory pool smaller than "
            f"{cfg.tpe_v2_min_trajs}; no offline stats this step.)"
        )
        out = (out.rstrip() + "\n\n" + note) if out else note
    return out


def restore_reward_file_to_best(
    *,
    specific_code_path: Optional[str],
    best_code: str,
    failed_code: str,
) -> bool:
    """Backup failed reward and overwrite on-disk specific file with best code."""
    if not specific_code_path:
        return False
    path = Path(specific_code_path)
    if not path.parent.exists():
        return False
    # specific_1.py -> specific_1.failed.py
    failed_path = path.with_name(path.stem + ".failed.py")
    try:
        failed_path.write_text(failed_code, encoding="utf-8")
        path.write_text(best_code, encoding="utf-8")
        return True
    except OSError as exc:
        logger.warning("CARD+ rollback file restore failed: %s", exc)
        return False


def on_policy_tpe_v2(
    cfg: CardPlusConfig,
    trajs: Sequence[Dict[str, Any]],
    *,
    pass_v1: Optional[bool],
    acc_v1: Optional[float],
    task_name: str,
    benchmark_name: str,
    gate_tpe: Optional[TPEv2Result],
) -> Optional[TPEv2Result]:
    """Recompute TPE v2 on the same traj pool used for sub-reward stats (after RL)."""
    if not cfg.use_tpe_v2:
        return gate_tpe
    if not trajs or len(trajs) < max(1, int(cfg.tpe_v2_min_trajs)):
        return gate_tpe
    return run_tpe_v2(
        trajs,
        pass_v1=bool(pass_v1) if pass_v1 is not None else True,
        acc_v1=float(acc_v1) if acc_v1 is not None else 0.0,
        cfg=cfg,
        task_name=task_name,
        reward_fn=None,
        use_stored_components=True,
        benchmark_name=benchmark_name,
    )


def update_archive_after_version(
    archive: RewardArchive,
    cfg: CardPlusConfig,
    *,
    iteration: int,
    reward_code: str,
    sub_reward_names: Optional[List[str]],
    rl_ran: bool,
    learning_result: Any,
    tpe_v2: Optional[TPEv2Result],
    pass_v1: Optional[bool],
    acc_v1: Optional[float],
    another_iteration_remains: bool,
) -> Tuple[VersionRecord, bool, str, Optional[str]]:
    """Register a version; optionally build a rollback notice.

    Returns (record, rolled_back, rollback_notice_or_empty, restored_best_code_or_none).
    """
    success_curve = extract_success_curve(learning_result) if rl_ran else []
    proxy_curve = extract_proxy_curve(learning_result) if rl_ran else []
    # Prefer env-grounded proxy (progress/soft-near/survive) so zero-success shorts still rank.
    score_curve = proxy_curve if proxy_curve else success_curve
    score = score_from_curve(score_curve, cfg.score_last_k_evals) if rl_ran else None
    auc = success_curve_auc(score_curve) if rl_ran and score_curve else None
    mean_ret = (
        extract_mean_return(learning_result, last_k=cfg.score_last_k_evals) if rl_ran else None
    )
    mean_prog = (
        extract_mean_progress(learning_result, last_k=cfg.score_last_k_evals) if rl_ran else None
    )

    # Decide rollback against previous best BEFORE adding current (if current is bad)
    rolled_back = False
    notice = ""
    restored_best_code: Optional[str] = None
    if (
        cfg.use_archive_rollback
        and rl_ran
        and score is not None
        and another_iteration_remains
        and archive.should_rollback_against(score, exclude_iteration=iteration)
    ):
        # Same peer set as the rollback decision (never restore the current iteration).
        best = archive.best_record_excluding(exclude_iteration=iteration)
        if best is not None:
            rolled_back = True
            restored_best_code = best.reward_code
            failed_preview = VersionRecord(
                iteration=iteration,
                reward_code=reward_code,
                sub_reward_names=sub_reward_names or [],
                rl_ran=True,
                success_curve=success_curve,
                proxy_curve=proxy_curve,
                version_score=score,
                curve_auc=auc,
                mean_return=mean_ret,
                mean_progress=mean_prog,
            )
            notice = archive.format_rollback_notice(failed_preview, best)

    record = VersionRecord(
        iteration=iteration,
        reward_code=reward_code,
        sub_reward_names=list(sub_reward_names or []),
        rl_ran=rl_ran,
        success_curve=success_curve,
        proxy_curve=proxy_curve,
        version_score=score,
        curve_auc=auc,
        mean_return=mean_ret,
        mean_progress=mean_prog,
        tpe_acc_v1=acc_v1,
        tpe_pass_v1=pass_v1,
        rho_full=tpe_v2.rho_full if tpe_v2 else None,
        rho_dense=tpe_v2.rho_dense if tpe_v2 else None,
        hacked=tpe_v2.hacked if tpe_v2 else None,
        verdict=tpe_v2.verdict if tpe_v2 else None,
    )
    if cfg.use_archive_rollback or cfg.any_enabled:
        archive.add(record)
        _persist_best_code(archive)

    return record, rolled_back, notice, restored_best_code


def _persist_best_code(archive: RewardArchive) -> None:
    if archive.run_dir is None:
        return
    best = archive.best_record()
    if best is None:
        return
    out_dir = Path(archive.run_dir) / "card_plus_best"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "specific.py").write_text(best.reward_code, encoding="utf-8")
    (out_dir / "version.txt").write_text(str(best.iteration), encoding="utf-8")


def write_iteration_log(
    run_dir: str,
    cfg: CardPlusConfig,
    *,
    iteration: int,
    tpe_v2: Optional[TPEv2Result],
    pass_v1: Optional[bool],
    acc_v1: Optional[float],
    rl_ran: bool,
    version_score: Optional[float],
    best_idx: Optional[int],
    rolled_back: bool,
    token_counts: Optional[Dict[str, Any]] = None,
    extra_score_final_version: bool = False,
    extra_train_cost: Optional[str] = None,
    diff_lines_vs_best: Optional[int] = None,
    diff_lines_vs_prev: Optional[int] = None,
    is_final_step: Optional[bool] = None,
) -> None:
    if not cfg.any_enabled:
        return
    append_card_plus_log(
        run_dir,
        make_iteration_log(
            iteration=iteration,
            flags={
                "use_archive_rollback": cfg.use_archive_rollback,
                "use_subreward_stats": cfg.use_subreward_stats,
                "use_tpe_v2": cfg.use_tpe_v2,
                "tpe_v2_mode": cfg.tpe_v2_mode,
            },
            tpe_acc_v1=acc_v1,
            tpe_pass_v1=pass_v1,
            rho_full=tpe_v2.rho_full if tpe_v2 else None,
            rho_dense=tpe_v2.rho_dense if tpe_v2 else None,
            sparse_keys=tpe_v2.sparse_keys if tpe_v2 else [],
            hacked=tpe_v2.hacked if tpe_v2 else None,
            verdict=tpe_v2.verdict if tpe_v2 else None,
            rl_ran=rl_ran,
            version_score=version_score,
            best_idx=best_idx,
            rolled_back=rolled_back,
            token_counts=token_counts,
            extra_score_final_version=extra_score_final_version,
            extra_train_cost=extra_train_cost,
            diff_lines_vs_best=diff_lines_vs_best,
            diff_lines_vs_prev=diff_lines_vs_prev,
            no_success_pool=tpe_v2.no_success_pool if tpe_v2 else None,
            pass_v1_overridden=tpe_v2.pass_v1_overridden if tpe_v2 else None,
            is_final_step=is_final_step,
        ),
    )


def maybe_score_final_version(
    cfg: CardPlusConfig,
    archive: RewardArchive,
    *,
    is_final_step: bool,
    train_fn: Callable[[str], Any],
) -> Optional[VersionRecord]:
    """If last version was never trained, optionally run the short protocol once.

    ``train_fn(code) -> learning_result`` should call the same train_and_eval path.
    """
    if not (cfg.use_archive_rollback and cfg.score_final_version and is_final_step):
        return None
    if not archive.versions:
        return None
    last = max(archive.versions, key=lambda v: v.iteration)
    if last.rl_ran and last.version_score is not None:
        return None
    cost_msg = (
        f"CARD+ extra training cost: one short-protocol train_and_eval for "
        f"unscored last version {last.iteration} (score_final_version)."
    )
    logger.info(cost_msg)
    print(cost_msg)
    learning_result = train_fn(last.reward_code)
    success_curve = extract_success_curve(learning_result)
    proxy_curve = extract_proxy_curve(learning_result)
    score_curve = proxy_curve if proxy_curve else success_curve
    last.rl_ran = True
    last.success_curve = success_curve
    last.proxy_curve = proxy_curve
    last.version_score = score_from_curve(score_curve, cfg.score_last_k_evals)
    last.curve_auc = success_curve_auc(score_curve) if score_curve else None
    last.mean_return = extract_mean_return(learning_result, last_k=cfg.score_last_k_evals)
    last.mean_progress = extract_mean_progress(learning_result, last_k=cfg.score_last_k_evals)
    archive.add(last)
    _persist_best_code(archive)
    return last


def _diff_sizes(
    archive: RewardArchive,
    reward_code: str,
    exclude_iteration: int,
) -> Tuple[Optional[int], Optional[int]]:
    others = [v for v in archive.versions if v.iteration != exclude_iteration]
    prev = max(others, key=lambda v: v.iteration) if others else None
    scored = [v for v in others if v.rl_ran and v.version_score is not None]
    best = None
    if scored:
        best = max(scored, key=archive._score_key)
    vs_best = unified_diff_line_count(best.reward_code, reward_code) if best else None
    vs_prev = unified_diff_line_count(prev.reward_code, reward_code) if prev else None
    return vs_best, vs_prev


def apply_card_plus_after_baseline(
    feedback_str: str,
    cfg: CardPlusConfig,
    *,
    archive: Optional[RewardArchive],
    iteration: int,
    reward_code: str,
    trajs: Sequence[Dict[str, Any]],
    task_name: str,
    benchmark_name: str,
    tpe_v2: Optional[TPEv2Result],
    rl_ran: bool,
    learning_result: Any,
    pass_v1: Optional[bool],
    acc_v1: Optional[float],
    run_dir: str,
    reward_fn: Optional[Callable[..., Any]] = None,
    train_final_fn: Optional[Callable[[str], Any]] = None,
    token_counts: Optional[Dict[str, Any]] = None,
    specific_code_path: Optional[str] = None,
    mutate_archive: bool = True,
) -> str:
    """Apply CARD+ after baseline process/trajectory/preference text.

    Order: process/trajectory → (dense-shaping) → stats → rollback notice.
    Stats header is labeled with this version index when a rollback notice is added.

    After RL, TPE v2 diagnostics are recomputed on the same on-policy traj pool
    used for sub-reward stats so feedback stays consistent. Pre-train ``tpe_v2``
    remains the gate decision in the caller.

    ``mutate_archive=False`` (checkpoint resume): enrich feedback only; do not
    re-register the version, re-rollback, or rewrite ``card_plus_best/``.
    """
    if not cfg.any_enabled:
        return feedback_str
    if archive is None:
        archive = RewardArchive(run_dir=run_dir, cfg=cfg)

    is_final = is_final_train_step(iteration, cfg.card_plus_max_step)
    another_left = another_coder_query_follows(iteration, cfg.card_plus_max_step)
    diff_best, diff_prev = _diff_sizes(archive, reward_code, exclude_iteration=iteration)

    # Align dense-shaping diagnostics with the trajs used for stats.
    feedback_tpe = tpe_v2
    if rl_ran:
        feedback_tpe = on_policy_tpe_v2(
            cfg,
            trajs,
            pass_v1=pass_v1,
            acc_v1=acc_v1,
            task_name=task_name,
            benchmark_name=benchmark_name,
            gate_tpe=tpe_v2,
        )

    rolled_back = False
    rollback_notice = ""
    restored_best = None
    record = None
    if mutate_archive:
        record, rolled_back, rollback_notice, restored_best = update_archive_after_version(
            archive,
            cfg,
            iteration=iteration,
            reward_code=reward_code,
            sub_reward_names=sub_reward_names_from_trajs(trajs),
            rl_ran=rl_ran,
            learning_result=learning_result,
            tpe_v2=feedback_tpe,
            pass_v1=pass_v1,
            acc_v1=acc_v1,
            another_iteration_remains=another_left,
        )

        if rolled_back and restored_best:
            restored = restore_reward_file_to_best(
                specific_code_path=specific_code_path,
                best_code=restored_best,
                failed_code=reward_code,
            )
            if restored:
                peer = archive.best_record_excluding(exclude_iteration=iteration)
                peer_idx = peer.iteration if peer is not None else archive.best_index()
                print(
                    f"CARD+ rollback: restored on-disk reward to best version "
                    f"{peer_idx} (failed saved alongside)."
                )
    else:
        # Read-only resume: synthesize a lightweight record for logging/enrichment.
        success_curve = extract_success_curve(learning_result) if rl_ran else []
        proxy_curve = extract_proxy_curve(learning_result) if rl_ran else []
        score_curve = proxy_curve if proxy_curve else success_curve
        record = VersionRecord(
            iteration=iteration,
            reward_code=reward_code,
            sub_reward_names=sub_reward_names_from_trajs(trajs),
            rl_ran=rl_ran,
            success_curve=success_curve,
            proxy_curve=proxy_curve,
            version_score=(
                score_from_curve(score_curve, cfg.score_last_k_evals) if rl_ran else None
            ),
            curve_auc=success_curve_auc(score_curve) if rl_ran and score_curve else None,
            mean_return=(
                extract_mean_return(learning_result, last_k=cfg.score_last_k_evals)
                if rl_ran
                else None
            ),
            mean_progress=(
                extract_mean_progress(learning_result, last_k=cfg.score_last_k_evals)
                if rl_ran
                else None
            ),
            tpe_acc_v1=acc_v1,
            tpe_pass_v1=pass_v1,
            rho_full=feedback_tpe.rho_full if feedback_tpe else None,
            rho_dense=feedback_tpe.rho_dense if feedback_tpe else None,
            hacked=feedback_tpe.hacked if feedback_tpe else None,
            verdict=feedback_tpe.verdict if feedback_tpe else None,
        )

    version_for_stats = iteration if rolled_back else None
    feedback_str = enrich_feedback(
        feedback_str,
        cfg,
        trajs=trajs,
        task_name=task_name,
        tpe_v2=feedback_tpe,
        rl_skipped=not rl_ran,
        reward_fn=reward_fn if not rl_ran else None,
        benchmark_name=benchmark_name,
        version_index=version_for_stats,
        version_score=record.version_score,
    )
    if rolled_back and rollback_notice:
        feedback_str = feedback_str.rstrip() + "\n\n" + rollback_notice

    extra_scored = None
    extra_cost = None
    if mutate_archive and is_final and train_final_fn is not None:
        extra_scored = maybe_score_final_version(
            cfg,
            archive,
            is_final_step=True,
            train_fn=train_final_fn,
        )
        if extra_scored is not None:
            extra_cost = (
                "one extra short-protocol train_and_eval "
                f"(score_final_version, version {extra_scored.iteration})"
            )
            print(extra_cost)
        if archive.best_code is not None:
            print(
                f"CARD+ best version is iteration {archive.best_index()} "
                f"(see {os.path.join(run_dir, 'card_plus_best')})."
            )

    if mutate_archive:
        write_iteration_log(
            run_dir,
            cfg,
            iteration=iteration,
            tpe_v2=feedback_tpe,
            pass_v1=pass_v1,
            acc_v1=acc_v1,
            rl_ran=rl_ran if extra_scored is None else True,
            version_score=record.version_score if extra_scored is None else extra_scored.version_score,
            best_idx=archive.best_index(),
            rolled_back=rolled_back,
            token_counts=token_counts,
            extra_score_final_version=extra_scored is not None,
            extra_train_cost=extra_cost,
            diff_lines_vs_best=diff_best,
            diff_lines_vs_prev=diff_prev,
            is_final_step=is_final,
        )
    return feedback_str


def sub_reward_names_from_trajs(trajs: Sequence[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    seen = set()
    for t in trajs:
        for k in t.keys():
            if isinstance(k, str) and k.startswith("sub_reward/"):
                name = k[len("sub_reward/") :]
                if name not in seen:
                    seen.add(name)
                    names.append(name)
    return names
