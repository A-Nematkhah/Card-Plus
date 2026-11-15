"""Dataset keep-gate thresholds (no simulator / no full dataset load)."""

from __future__ import annotations

from card_plus.dataset.gate import DatasetGateThresholds, evaluate_keep
from card_plus.dataset.rescore import PairAccuracy, RescoreReport


def _report(a: float, b: float, c: float, auc: float) -> RescoreReport:
    return RescoreReport(
        n_episodes=10,
        mean_return=0.0,
        outcome_counts={"success": 4, "collided": 3, "timeout": 3},
        progress_auc=auc,
        tpe_v1_rate=0.5,
        pair_accuracies=[
            PairAccuracy("success/collided", 12, a),
            PairAccuracy("success/timeout", 12, b),
            PairAccuracy("timeout/collided", 9, c),
        ],
    )


def test_gt_like_metrics_keep():
    keep, reasons, score, _ = evaluate_keep(_report(1.0, 1.0, 1.0, 1.0))
    assert keep is True
    assert reasons == []
    assert score >= 0.75


def test_sparse_hacking_rejected_by_timeout_collided():
    keep, reasons, _, _ = evaluate_keep(_report(1.0, 1.0, 0.0, 1.0))
    assert keep is False
    assert any("timeout/collided" in r for r in reasons)


def test_progress_like_rejected_by_timeout_collided():
    keep, reasons, _, _ = evaluate_keep(_report(0.99, 0.99, 0.39, 0.99))
    assert keep is False
    assert any("timeout/collided" in r for r in reasons)


def test_specific0_borderline_passes_optimal_thresholds():
    # Observed smoke LLM reward on crowdnav_ds_v1
    keep, reasons, score, metrics = evaluate_keep(
        _report(0.9248, 0.8662, 0.6710, 0.8789)
    )
    assert keep is True, reasons
    assert metrics["composite"] == score
    assert score >= 0.75


def test_evaluate_keep_uses_offline_rank_floor_not_calibrated_min():
    # Pair floors pass, but offline_rank floor (0.90) rejects.
    th = DatasetGateThresholds(min_composite=0.65, min_offline_rank=0.90)
    keep, reasons, score, _ = evaluate_keep(
        _report(0.85, 0.80, 0.60, 0.75), th
    )
    assert keep is False
    assert 0.75 <= score < 0.90
    assert any("composite=" in r for r in reasons)


def test_custom_harder_timeout_collided_floor():
    th = DatasetGateThresholds(min_timeout_collided=0.70)
    keep, reasons, _, _ = evaluate_keep(_report(0.92, 0.87, 0.67, 0.88), th)
    assert keep is False
    assert any("timeout/collided" in r for r in reasons)
