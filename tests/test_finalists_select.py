"""Unit tests for across-step SH finalist selection."""

from card_plus.archive import VersionRecord
from code_generation.self_reflection.crowdnav_finalists import select_top_versions


def _v(iteration: int, score: float, peak: float, rl: bool = True, auc: float = None) -> VersionRecord:
    return VersionRecord(
        iteration=iteration,
        reward_code=f"code_{iteration}",
        rl_ran=rl,
        success_curve=[0.0, peak],
        version_score=score,
        curve_auc=peak if auc is None else auc,
    )


def test_select_top_two_by_score_when_spread_large():
    versions = [
        _v(1, 0.1, 0.2),
        _v(2, 0.4, 0.5),
        _v(3, 0.3, 0.6),
        _v(4, 0.0, 0.0, rl=False),
    ]
    top = select_top_versions(versions, top_k=2)
    # max(score,peak): iter3=0.6, iter2=0.5
    assert [v.iteration for v in top] == [3, 2]


def test_select_skips_empty_code():
    versions = [
        VersionRecord(iteration=1, reward_code="", rl_ran=True, version_score=0.9),
        _v(2, 0.2, 0.3),
    ]
    top = select_top_versions(versions, top_k=2)
    assert [v.iteration for v in top] == [2]


def test_low_score_spread_prefers_peak():
    # All version_scores nearly equal (H=10 style); peak should decide.
    versions = [
        _v(1, 0.08, 0.15),
        _v(2, 0.07, 0.30),
        _v(3, 0.08, 0.15),
        _v(4, 0.02, 0.25),
    ]
    top = select_top_versions(versions, top_k=2)
    assert top[0].iteration == 2
    assert top[1].iteration == 4
