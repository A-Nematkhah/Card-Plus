"""Tests for audit fixes: final scoring, last-iter rollback, order, sparse subtract."""

from __future__ import annotations

import numpy as np

from card_plus.archive import RewardArchive, VersionRecord, unified_diff_line_count
from card_plus.config import CardPlusConfig
from card_plus.hooks import apply_card_plus_after_baseline, maybe_score_final_version
from card_plus.loop_control import another_coder_query_follows, is_final_train_step
from card_plus.reward_utils import dense_totals, mean_per_step_return


def test_final_step_from_train_rounds_not_reflection_num():
    assert is_final_train_step(3, 3) is True
    assert is_final_train_step(2, 3) is False
    assert is_final_train_step(10, 3) is True
    assert is_final_train_step(10, None) is False  # unknown horizon: do not guess
    assert another_coder_query_follows(3, 3) is False
    assert another_coder_query_follows(2, 3) is True


def test_score_final_version_trains_unscored_on_last_step(tmp_path):
    cfg = CardPlusConfig(use_archive_rollback=True, score_final_version=True, card_plus_max_step=3)
    arch = RewardArchive(run_dir=str(tmp_path), cfg=cfg)
    arch.add(VersionRecord(iteration=3, reward_code="code_v3", rl_ran=False, version_score=None))
    calls = []

    def train_fn(code: str):
        calls.append(code)
        return {"successes": np.array([0.2, 0.4, 0.6])}

    scored = maybe_score_final_version(cfg, arch, is_final_step=True, train_fn=train_fn)
    assert scored is not None
    assert calls == ["code_v3"]
    assert scored.rl_ran is True
    assert scored.version_score is not None
    assert arch.best_index() == 3


def test_score_final_version_skipped_when_not_last_step(tmp_path):
    cfg = CardPlusConfig(use_archive_rollback=True, score_final_version=True, card_plus_max_step=3)
    arch = RewardArchive(run_dir=str(tmp_path), cfg=cfg)
    arch.add(VersionRecord(iteration=2, reward_code="code_v2", rl_ran=False))
    scored = maybe_score_final_version(
        cfg, arch, is_final_step=False, train_fn=lambda c: (_ for _ in ()).throw(AssertionError("should not train"))
    )
    assert scored is None


def test_no_rollback_on_last_iteration(tmp_path):
    cfg = CardPlusConfig(
        use_archive_rollback=True,
        use_subreward_stats=True,
        rollback_margin=0.05,
        card_plus_max_step=3,
        score_last_k_evals=1,
    )
    arch = RewardArchive(run_dir=str(tmp_path), cfg=cfg)
    arch.add(
        VersionRecord(
            iteration=1,
            reward_code="best_code",
            rl_ran=True,
            success_curve=[0.8],
            version_score=0.8,
            curve_auc=0.8,
        )
    )
    fb = "1.process\n2.trajectory"
    out = apply_card_plus_after_baseline(
        fb,
        cfg,
        archive=arch,
        iteration=3,
        reward_code="worse_code",
        trajs=[],
        task_name="drawer-open-v2",
        benchmark_name="metaworld",
        tpe_v2=None,
        rl_ran=True,
        learning_result={"successes": np.array([0.1])},
        pass_v1=True,
        acc_v1=1.0,
        run_dir=str(tmp_path),
        train_final_fn=None,
    )
    assert "# Rollback Notice" not in out


def test_stats_then_rollback_order_and_version_label(tmp_path):
    cfg = CardPlusConfig(
        use_archive_rollback=True,
        use_subreward_stats=True,
        rollback_margin=0.05,
        card_plus_max_step=5,
        score_last_k_evals=1,
        max_feedback_lines=40,
    )
    arch = RewardArchive(run_dir=str(tmp_path), cfg=cfg)
    arch.add(
        VersionRecord(
            iteration=1,
            reward_code="def best():\n    return 1\n",
            rl_ran=True,
            success_curve=[0.9],
            version_score=0.9,
        )
    )
    t = 8
    xs = np.linspace(0, 0.9, t)
    obs = np.zeros((t, 39))
    obs[:, 4] = xs
    traj = {
        "obs": obs,
        "actions": np.zeros((t, 4)),
        "rewards": np.ones(t),
        "dones": np.zeros(t, dtype=bool),
        "infos": [{"success": True}] * t,
        "self.obj1.position": np.stack([xs, np.zeros(t), np.zeros(t)], axis=1),
        "self.goal_position": np.tile([1.0, 0.0, 0.0], (t, 1)),
        "sub_reward/a": np.ones(t),
    }
    process = "1.process feedback\n2.trajectory feedback"
    out = apply_card_plus_after_baseline(
        process,
        cfg,
        archive=arch,
        iteration=2,
        reward_code="def worse():\n    return 0\n",
        trajs=[traj],
        task_name="drawer-open-v2",
        benchmark_name="metaworld",
        tpe_v2=None,
        rl_ran=True,
        learning_result={"successes": np.array([0.1])},
        pass_v1=True,
        acc_v1=1.0,
        run_dir=str(tmp_path),
    )
    i_proc = out.find("1.process feedback")
    i_stats = out.find("# Sub-reward Statistics")
    i_rb = out.find("# Rollback Notice")
    assert 0 <= i_proc < i_stats < i_rb
    assert "for reward version 2" in out


def test_sparse_subtraction_when_total_neq_sum_components():
    totals = np.array([10.0, 10.0, 10.0])  # residual 7 not in any component
    comps = {"sparse": np.array([3.0, 0.0, 0.0]), "dense": np.array([0.0, 0.0, 0.0])}
    dense = dense_totals(totals, comps, ["sparse"])
    assert np.allclose(dense, np.array([7.0, 10.0, 10.0]))
    assert mean_per_step_return(dense) == (7.0 + 10.0 + 10.0) / 3.0
    # re-sum would have been 0+0+0; we must NOT do that
    assert not np.allclose(dense, comps["dense"])


def test_diff_line_counts_logged(tmp_path):
    cfg = CardPlusConfig(use_archive_rollback=True, card_plus_max_step=5)
    arch = RewardArchive(run_dir=str(tmp_path), cfg=cfg)
    arch.add(VersionRecord(iteration=1, reward_code="aaa\nbbb\n", rl_ran=True, version_score=0.5, success_curve=[0.5]))
    apply_card_plus_after_baseline(
        "1.x\n2.y",
        cfg,
        archive=arch,
        iteration=2,
        reward_code="aaa\nccc\n",
        trajs=[],
        task_name="drawer-open-v2",
        benchmark_name="metaworld",
        tpe_v2=None,
        rl_ran=True,
        learning_result={"successes": np.array([0.4])},
        pass_v1=True,
        acc_v1=1.0,
        run_dir=str(tmp_path),
    )
    log = (tmp_path / "card_plus_log.jsonl").read_text(encoding="utf-8")
    assert "diff_lines_vs_best" in log
    assert "diff_lines_vs_prev" in log
    assert unified_diff_line_count("aaa\nbbb\n", "aaa\nccc\n") > 0


def test_eval_best_rejects_same_seed(tmp_path):
    from card_plus.eval_best import prepare_final_eval

    d = tmp_path / "card_plus_best"
    d.mkdir()
    (d / "specific.py").write_text("def compute_dense_reward():\n    pass\n", encoding="utf-8")
    (d / "version.txt").write_text("2", encoding="utf-8")
    try:
        prepare_final_eval(str(tmp_path), eval_seed=1, selection_seed=1)
        assert False, "expected ValueError"
    except ValueError:
        pass
    info = prepare_final_eval(str(tmp_path), eval_seed=1001, selection_seed=1)
    assert info["eval_seed"] == 1001
    assert "compute_dense_reward" in info["code"]
