"""Section-3 fixes: threshold fallbacks, NaN floors, gate json reload, intra NaN."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from card_plus.dataset.gate import (
    DatasetGateThresholds,
    dataset_gate_from_args,
    evaluate_pair_floors,
    load_dataset_gate_result_json,
)
from card_plus.dataset.rescore import PairAccuracy, RescoreReport
from card_plus.dataset.scoring import calibrated_composite, intra_scenario_score
from card_plus.archive import VersionRecord
from code_generation.self_reflection.crowdnav_finalists import select_top_versions


def test_dataset_gate_from_args_matches_threshold_defaults():
    cfg = dataset_gate_from_args(Namespace())
    defaults = DatasetGateThresholds()
    assert cfg.thresholds.min_composite == defaults.min_composite
    assert cfg.thresholds.min_anti_hack == defaults.min_anti_hack
    assert cfg.thresholds.min_separation == defaults.min_separation
    assert cfg.thresholds.min_composite == 0.75
    assert cfg.thresholds.min_anti_hack == 0.65
    assert cfg.thresholds.min_separation == 0.70


def test_nan_pair_floor_fails_keep():
    report = RescoreReport(
        n_episodes=5,
        mean_return=0.0,
        outcome_counts={"success": 5},
        progress_auc=1.0,
        pair_accuracies=[
            PairAccuracy("success/collided", 0, float("nan")),
            PairAccuracy("success/timeout", 0, float("nan")),
            PairAccuracy("timeout/collided", 0, float("nan")),
        ],
    )
    ok, reasons, _, _ = evaluate_pair_floors(report)
    assert ok is False
    assert any("success/collided=nan" in r for r in reasons)


def test_intra_undefined_is_nan_and_keeps_weight_as_zero():
    m = intra_scenario_score([], [])
    assert m["intra_n_scenarios"] == 0.0
    assert m["intra_scenario"] != m["intra_scenario"]  # NaN
    # NaN intra is treated as 0.0 with weight kept (no renormalization inflate).
    with_nan = calibrated_composite(
        offline_rank=0.9, separation=0.8, anti_hack=0.8, intra_scenario=float("nan")
    )
    as_zero = calibrated_composite(
        offline_rank=0.9, separation=0.8, anti_hack=0.8, intra_scenario=0.0
    )
    assert abs(with_nan - as_zero) < 1e-12
    # Dropping the term would inflate; keep-weight must stay below that.
    dropped = (0.9 * 0.10 + 0.8 * 0.30 + 0.8 * 0.30) / (0.10 + 0.30 + 0.30)
    assert with_nan < dropped


def test_load_dataset_gate_result_json_roundtrip(tmp_path: Path):
    payload = {
        "keep": True,
        "skip_rl": False,
        "mode": "gate",
        "composite": 0.81,
        "reasons": [],
        "metrics": {"offline_rank": 0.9, "separation": 0.8, "anti_hack": 0.7, "intra_scenario": 0.75},
        "report": {
            "n_episodes": 12,
            "mean_return": 1.0,
            "outcome_counts": {"success": 4, "collided": 4, "timeout": 4},
            "progress_auc": 0.88,
            "tpe_v1_rate": 0.9,
            "pair_accuracies": [
                {"pair": "success/collided", "n_pairs": 16, "accuracy": 0.9},
            ],
            "extras": {"outcome_return_auc": 0.88},
        },
    }
    path = tmp_path / "dataset_gate_0.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_dataset_gate_result_json(path, mode="gate")
    assert loaded is not None
    assert loaded.keep is True
    assert loaded.skip_rl is False
    assert abs(loaded.composite - 0.81) < 1e-9
    assert loaded.report.n_episodes == 12
    assert loaded.metrics["offline_rank"] == 0.9


def test_finalists_prefer_higher_mean_return_on_tie():
    versions = [
        VersionRecord(
            iteration=1,
            reward_code="a",
            rl_ran=True,
            version_score=1.0,
            success_curve=[1.0],
            curve_auc=0.6,
            mean_return=22.0,
        ),
        VersionRecord(
            iteration=2,
            reward_code="b",
            rl_ran=True,
            version_score=1.0,
            success_curve=[1.0],
            curve_auc=0.6,
            mean_return=11.0,
        ),
    ]
    top = select_top_versions(versions, top_k=1)
    assert top[0].iteration == 1
