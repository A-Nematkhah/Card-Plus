"""Shared CARD one-step protocol used by tests (mirrors both *_exp_one_step.py files).

When all CARD+ flags are False this is git 6e1df1c: always train, then
``assemble_baseline_feedback``. The real scripts call the same helpers.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from card_plus.baseline_feedback import assemble_baseline_feedback
from card_plus.config import CardPlusConfig
from card_plus.hooks import apply_card_plus_after_baseline
from card_plus.tpe_v2 import TPEv2Result, run_tpe_v2, should_skip_rl_for_tpe_v2


def simulate_one_step(
    *,
    script: str,
    cfg: CardPlusConfig,
    pass_flag: bool,
    acc_v1: float,
    check_feedback: str,
    learning_result_feedback: str,
    eval_feedback: str,
    use_return_check: bool = True,
    exclude: Optional[str] = None,
    train_fn: Optional[Callable[[], Any]] = None,
    trajs: Optional[Sequence[Dict[str, Any]]] = None,
    reward_code: str = "def compute_dense_reward(self, action, obs):\n    return 0.0, {}\n",
    iteration: int = 1,
    run_dir: str,
    task_name: str,
    benchmark_name: str,
    learning_result: Any = None,
) -> Tuple[List[str], str]:
    """Return (call_sequence, feedback_str). ``script`` is recorded in the log only."""
    del script  # both scripts share this protocol
    calls: List[str] = []
    tpe_v2: Optional[TPEv2Result] = None
    if use_return_check:
        calls.append("trajectories_return_check")
        if cfg.use_tpe_v2 and trajs:
            tpe_v2 = run_tpe_v2(
                trajs,
                pass_v1=pass_flag,
                acc_v1=acc_v1,
                cfg=cfg,
                task_name=task_name,
                benchmark_name=benchmark_name,
                use_stored_components=True,
            )
            calls.append("tpe_v2")
    skip = bool(tpe_v2 is not None and should_skip_rl_for_tpe_v2(cfg, tpe_v2))
    rl_ran = False
    result = learning_result
    if skip:
        calls.append("skip_rl")
    else:
        if train_fn is not None:
            result = train_fn()
        calls.append("train_and_eval")
        rl_ran = True
    effective_pass = pass_flag
    if (not rl_ran) and cfg.use_tpe_v2 and cfg.tpe_v2_mode == "gate":
        effective_pass = False
    feedback, _extend = assemble_baseline_feedback(
        use_return_check=use_return_check,
        pass_flag=effective_pass,
        exclude=exclude,
        learning_result_feedback=learning_result_feedback,
        eval_feedback=eval_feedback,
        check_feedback=check_feedback,
    )
    feedback = apply_card_plus_after_baseline(
        feedback,
        cfg,
        archive=None,
        iteration=iteration,
        reward_code=reward_code,
        trajs=list(trajs or []),
        task_name=task_name,
        benchmark_name=benchmark_name,
        tpe_v2=tpe_v2,
        rl_ran=rl_ran,
        learning_result=result,
        pass_v1=tpe_v2.pass_v1 if tpe_v2 is not None else pass_flag,
        acc_v1=acc_v1,
        run_dir=run_dir,
        train_final_fn=None,
    )
    return calls, feedback
