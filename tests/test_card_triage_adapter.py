"""Tests for optional reward_triage bridge (flags default False)."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from card_plus.baseline_feedback import assemble_baseline_feedback
from card_plus.config import CardPlusConfig
from card_plus.one_step_sim import simulate_one_step
from card_triage_adapter import (
    TriageFlags,
    after_iteration,
    before_train,
    triage_flags_from_args,
)

_REPO = Path(__file__).resolve().parents[1]
_MW = _REPO / "code_generation" / "self_reflection" / "metaworld_exp_one_step.py"
_CN = _REPO / "code_generation" / "self_reflection" / "crowdnav_exp_one_step.py"

_PROCESS = "Evaluation is conducted at specific training intervals.\n"
_TRAJ = "Trajectory log here.\n"
_PREF = "We collected several trajectories where some agents successfully solved the task.\n"

_GOOD_CODE = (
    "def compute_dense_reward(self, action, obs):\n"
    "    import numpy as np\n"
    "    r = float(np.sum(obs) * 0.01 + np.sum(action) * 0.001)\n"
    "    return r, {'term': r}\n"
)
_BAD_CONST = (
    "def compute_dense_reward(self, action, obs):\n"
    "    return 0.0, {}\n"
)


def _args(**kwargs):
    base = dict(
        benchmark_name="metaworld",
        task_name="drawer-open-v2",
        max_episode_steps=500,
        train_max_steps=1000,
        eval_freq=100,
        train_num=1,
        n_eval_episodes=10,
        seed=1,
        train_seed_num=1,
        triage_validate=False,
        triage_cache=False,
        triage_plateau_stop=False,
        triage_shadow=False,
        triage_plateau_margin=2.0,
        triage_plateau_patience=1,
        triage_cache_root=None,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_flags_default_false():
    flags = triage_flags_from_args(Namespace())
    assert flags.any_enabled is False


def test_scripts_gate_triage_and_keep_baseline_helpers():
    for path in (_MW, _CN):
        text = path.read_text(encoding="utf-8")
        assert "add_triage_arguments" in text
        assert "triage_flags.any_enabled" in text
        assert "assemble_baseline_feedback" in text
        assert "apply_card_plus_after_baseline" in text


def test_golden_flags_off_unchanged(tmp_path):
    """Triage flags off: same as CARD+ golden (always train + baseline feedback)."""
    cfg = CardPlusConfig()
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

    calls, fb = simulate_one_step(
        script="metaworld_exp_one_step.py",
        cfg=cfg,
        pass_flag=False,
        acc_v1=0.1,
        check_feedback=_PREF,
        learning_result_feedback=_PROCESS,
        eval_feedback=_TRAJ,
        train_fn=train,
        run_dir=str(tmp_path / "golden"),
        task_name="drawer-open-v2",
        benchmark_name="metaworld",
    )
    assert calls == ["trajectories_return_check", "train_and_eval"]
    assert fb == expected_fb
    assert train_calls == ["train"]
    # Adapter no-op when flags off
    pre = before_train(
        TriageFlags(),
        reward_code=_GOOD_CODE,
        args=_args(),
        run_dir=str(tmp_path / "golden"),
    )
    assert pre.skip_rl is False
    assert pre.cache_hit is False


def test_triage_validate_rejects_constant(tmp_path):
    pytest.importorskip("reward_triage")
    flags = TriageFlags(triage_validate=True)
    pre = before_train(
        flags,
        reward_code=_BAD_CONST,
        args=_args(triage_validate=True),
        run_dir=str(tmp_path),
        train_max_steps=5000,
    )
    assert pre.validate_reject is True
    assert pre.skip_rl is True
    assert pre.steps_saved_est == 5000


def test_triage_cache_hit_skips_rl(tmp_path):
    pytest.importorskip("reward_triage")
    flags = TriageFlags(triage_cache=True)
    args = _args(triage_cache=True, train_max_steps=2000)
    # First: miss path via after_iteration put
    after_iteration(
        flags,
        reward_code=_GOOD_CODE,
        args=args,
        run_dir=str(tmp_path),
        iteration=1,
        rl_ran=True,
        learning_result={"successes": [0.1, 0.2, 0.3]},
        success_score=0.3,
        train_max_steps=2000,
    )
    pre = before_train(
        flags,
        reward_code=_GOOD_CODE,
        args=args,
        run_dir=str(tmp_path),
        train_max_steps=2000,
    )
    assert pre.cache_hit is True
    assert pre.skip_rl is True
    assert pre.steps_saved_est >= 2000


def test_triage_plateau_stop(tmp_path):
    pytest.importorskip("reward_triage")
    flags = TriageFlags(triage_plateau_stop=True, plateau_margin=2.0, plateau_patience=1)
    args = _args(triage_plateau_stop=True)
    p1 = after_iteration(
        flags,
        reward_code=_GOOD_CODE,
        args=args,
        run_dir=str(tmp_path),
        iteration=1,
        rl_ran=True,
        learning_result={"successes": [0.5]},
        success_score=0.5,
        n_eval_episodes=100,
        archive_rollback_enabled=False,
    )
    assert p1.should_stop is False
    p2 = after_iteration(
        flags,
        reward_code=_GOOD_CODE,
        args=args,
        run_dir=str(tmp_path),
        iteration=2,
        rl_ran=True,
        learning_result={"successes": [0.5]},
        success_score=0.5,
        n_eval_episodes=100,
        archive_rollback_enabled=False,
    )
    assert p2.should_stop is True
    assert (tmp_path / "triage_plateau_stop.json").exists()


def test_triage_shadow_logs_score(tmp_path):
    pytest.importorskip("reward_triage")
    flags = TriageFlags(triage_shadow=True)
    post = after_iteration(
        flags,
        reward_code=_GOOD_CODE,
        args=_args(triage_shadow=True),
        run_dir=str(tmp_path),
        iteration=1,
        rl_ran=True,
        learning_result={"successes": [0.7]},
        success_score=0.7,
    )
    assert post.shadow_score == 0.7
    assert post.should_stop is False
    log = (tmp_path / "triage_log.jsonl").read_text(encoding="utf-8")
    assert "shadow_score" in log


def test_smoke_all_flags_on(tmp_path):
    """Mock loop: validate + cache + plateau + shadow; no real SAC."""
    pytest.importorskip("reward_triage")
    flags = TriageFlags(
        triage_validate=True,
        triage_cache=True,
        triage_plateau_stop=True,
        triage_shadow=True,
        plateau_patience=1,
    )
    args = _args(
        triage_validate=True,
        triage_cache=True,
        triage_plateau_stop=True,
        triage_shadow=True,
        train_max_steps=1000,
    )
    run_dir = str(tmp_path / "smoke")
    # Reject bad code
    bad = before_train(
        flags, reward_code=_BAD_CONST, args=args, run_dir=run_dir, train_max_steps=1000
    )
    assert bad.skip_rl and bad.validate_reject
    # Good code trains (cache miss), then cache hit
    good = before_train(
        flags, reward_code=_GOOD_CODE, args=args, run_dir=run_dir, train_max_steps=1000
    )
    assert good.skip_rl is False or good.validate_reject is False
    after_iteration(
        flags,
        reward_code=_GOOD_CODE,
        args=args,
        run_dir=run_dir,
        iteration=1,
        rl_ran=True,
        learning_result={"successes": [0.4, 0.5]},
        success_score=0.5,
        n_eval_episodes=50,
        archive_rollback_enabled=False,
        train_max_steps=1000,
    )
    hit = before_train(
        flags, reward_code=_GOOD_CODE, args=args, run_dir=run_dir, train_max_steps=1000
    )
    assert hit.cache_hit is True
    # Plateau on flat scores
    after_iteration(
        flags,
        reward_code=_GOOD_CODE,
        args=args,
        run_dir=run_dir,
        iteration=2,
        rl_ran=False,
        learning_result={"successes": [0.5]},
        success_score=0.5,
        n_eval_episodes=50,
        archive_rollback_enabled=False,
    )
    assert (Path(run_dir) / "triage_log.jsonl").exists()
