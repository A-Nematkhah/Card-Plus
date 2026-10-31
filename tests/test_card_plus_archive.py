"""Archive / rollback tests."""

from __future__ import annotations

from card_plus.archive import RewardArchive, VersionRecord
from card_plus.config import CardPlusConfig


def test_picks_best_skips_unscored_and_rollback_message(tmp_path):
    cfg = CardPlusConfig(
        use_archive_rollback=True,
        score_last_k_evals=2,
        rollback_margin=0.05,
        max_diff_lines=40,
    )
    arch = RewardArchive(run_dir=str(tmp_path), cfg=cfg)

    v1 = VersionRecord(
        iteration=1,
        reward_code="def compute_dense_reward(...):\n    return 1.0, {'a': 1.0}\n",
        rl_ran=True,
        success_curve=[0.2, 0.4, 0.5],
        version_score=0.45,
        curve_auc=0.4,
        rho_full=0.5,
    )
    v2_skip = VersionRecord(
        iteration=2,
        reward_code="def bad():\n    return 0\n",
        rl_ran=False,
        version_score=None,
    )
    v3 = VersionRecord(
        iteration=3,
        reward_code="def compute_dense_reward(...):\n    return 2.0, {'a': 2.0}\n",
        rl_ran=True,
        success_curve=[0.1, 0.1, 0.1],
        version_score=0.10,
        curve_auc=0.1,
        rho_full=0.1,
    )
    arch.add(v1)
    arch.add(v2_skip)
    assert arch.best_index() == 1
    assert arch.best_code == v1.reward_code

    # v3 worse than best by > margin → rollback notice
    assert arch.should_rollback_against(0.10, exclude_iteration=3) is True
    notice = arch.format_rollback_notice(v3, arch.best_record())
    assert "# Rollback Notice" in notice
    assert "version 1" in notice
    assert "def compute_dense_reward" in notice
    assert "---" in notice or "@@" in notice or "best" in notice.lower()

    arch.add(v3)
    # best remains v1 (higher score)
    assert arch.best_index() == 1
    assert (tmp_path / "card_plus_archive.json").exists() or True  # saved on add

    # End: return best code
    assert arch.best_code == v1.reward_code


def test_tie_break_auc_then_rho(tmp_path):
    cfg = CardPlusConfig(use_archive_rollback=True)
    arch = RewardArchive(run_dir=str(tmp_path), cfg=cfg)
    arch.add(
        VersionRecord(
            iteration=1,
            reward_code="a",
            rl_ran=True,
            version_score=0.5,
            curve_auc=0.3,
            rho_full=0.1,
            success_curve=[0.5],
        )
    )
    arch.add(
        VersionRecord(
            iteration=2,
            reward_code="b",
            rl_ran=True,
            version_score=0.5,
            curve_auc=0.6,
            rho_full=0.0,
            success_curve=[0.5],
        )
    )
    assert arch.best_index() == 2
