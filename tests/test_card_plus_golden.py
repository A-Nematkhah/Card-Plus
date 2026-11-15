"""Golden regression: flags False == git 6e1df1c CARD loop (always train + same feedback)."""

from __future__ import annotations

from pathlib import Path

from card_plus.baseline_feedback import assemble_baseline_feedback
from card_plus.config import CardPlusConfig
from card_plus.one_step_sim import simulate_one_step

_REPO = Path(__file__).resolve().parents[1]
_MW = _REPO / "code_generation" / "self_reflection" / "metaworld_exp_one_step.py"
_CN = _REPO / "code_generation" / "self_reflection" / "crowdnav_exp_one_step.py"

# Frozen strings matching CARD git 6e1df1c assembly.
_PROCESS = "Evaluation is conducted at specific training intervals.\n"
_TRAJ = "Trajectory log here.\n"
_PREF = "We collected several trajectories where some agents successfully solved the task.\n"


def test_scripts_call_shared_6e1df1c_helpers():
    for path in (_MW, _CN):
        text = path.read_text(encoding="utf-8")
        assert "assemble_baseline_feedback" in text, path.name
        assert "apply_card_plus_after_baseline" in text, path.name
        assert "args.step >= args.reflection_num" not in text, path.name
        assert "card_plus_cfg.any_enabled" in text, path.name
        assert "triage_flags" not in text, path.name
        assert "add_triage_arguments" not in text, path.name
        assert "card_triage_adapter" not in text, path.name


def test_6e1df1c_feedback_pass_and_fail():
    fb, extend = assemble_baseline_feedback(
        use_return_check=True,
        pass_flag=True,
        exclude=None,
        learning_result_feedback=_PROCESS,
        eval_feedback=_TRAJ,
        check_feedback=_PREF,
    )
    assert fb == "1." + _PROCESS + "\n" + "2." + _TRAJ
    assert extend is True

    fb2, extend2 = assemble_baseline_feedback(
        use_return_check=True,
        pass_flag=False,
        exclude=None,
        learning_result_feedback=_PROCESS,
        eval_feedback=_TRAJ,
        check_feedback=_PREF,
    )
    assert fb2 == _PREF
    assert extend2 is False

    fb3, _ = assemble_baseline_feedback(
        use_return_check=True,
        pass_flag=False,
        exclude="preference",
        learning_result_feedback=_PROCESS,
        eval_feedback=_TRAJ,
        check_feedback=_PREF,
    )
    assert fb3 == "1." + _PROCESS + "\n" + "2." + _TRAJ


def test_flags_false_metaworld_and_crowdnav_identical_to_6e1df1c(tmp_path):
    cfg = CardPlusConfig()  # all False
    expected_fb, _ = assemble_baseline_feedback(
        use_return_check=True,
        pass_flag=False,
        exclude=None,
        learning_result_feedback=_PROCESS,
        eval_feedback=_TRAJ,
        check_feedback=_PREF,
    )
    train_calls = []

    def train():
        train_calls.append("train")
        return {"successes": [0.0]}

    for script, task, bench in (
        ("metaworld_exp_one_step.py", "drawer-open-v2", "metaworld"),
        ("crowdnav_exp_one_step.py", "circle-crossing-v0", "crowdnav"),
    ):
        train_calls.clear()
        calls, fb = simulate_one_step(
            script=script,
            cfg=cfg,
            pass_flag=False,  # TPE v1 fail — stock CARD still trains
            acc_v1=0.1,
            check_feedback=_PREF,
            learning_result_feedback=_PROCESS,
            eval_feedback=_TRAJ,
            train_fn=train,
            run_dir=str(tmp_path / script),
            task_name=task,
            benchmark_name=bench,
        )
        assert calls == ["trajectories_return_check", "train_and_eval"], script
        assert fb == expected_fb, script
        assert train_calls == ["train"], script
        assert "# Dense-Shaping Check" not in fb
        assert "# Sub-reward Statistics" not in fb
        assert "# Rollback Notice" not in fb
        assert not (tmp_path / script / "card_plus_log.jsonl").exists()
