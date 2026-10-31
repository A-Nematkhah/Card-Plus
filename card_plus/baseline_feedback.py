"""Original CARD (git 6e1df1c) feedback assembly. Used when CARD+ flags are False.

Stock CARD always trains after TPE; this helper only builds the Evaluator string
and says whether the new rollouts should be added to the stored pool.
"""

from __future__ import annotations

from typing import Optional, Tuple


def assemble_baseline_feedback(
    *,
    use_return_check: bool,
    pass_flag: bool,
    exclude: Optional[str],
    learning_result_feedback: str,
    eval_feedback: str,
    check_feedback: str,
) -> Tuple[str, bool]:
    """Return (feedback_str, extend_traj_pool) matching CARD git 6e1df1c.

    ``extend_traj_pool`` is True on the pass / no-check path (original CARD
    only grows the stored pool when TPE passes or is disabled).
    """
    feedback_str = ""
    if not use_return_check or pass_flag:
        if exclude == "process":
            feedback_str += eval_feedback
        elif exclude == "trajectory":
            feedback_str += learning_result_feedback
        else:
            feedback_str += "1." + learning_result_feedback + "\n"
            feedback_str += "2." + eval_feedback
        return feedback_str, True
    if exclude == "preference":
        feedback_str += "1." + learning_result_feedback + "\n"
        feedback_str += "2." + eval_feedback
    else:
        feedback_str += check_feedback
    return feedback_str, False
