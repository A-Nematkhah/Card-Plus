"""G2 learned judge unit tests."""

from __future__ import annotations

import numpy as np

from card_plus.dataset.learned_judge import fit_learned_judge, redesigned_composite_weights
from card_plus.dataset.scoring import calibrated_composite


def test_learned_judge_recovers_signed_signal():
    # anti_hack aligns with y; intra is anti-aligned — judge should upweight hack.
    metrics = []
    targets = []
    rng = np.random.default_rng(0)
    for i in range(16):
        hack = float(i) / 15.0
        intra = 1.0 - hack + float(rng.normal(0, 0.02))
        metrics.append(
            {
                "offline_rank": 0.9,
                "separation": 0.9,
                "anti_hack": hack,
                "intra_scenario": float(np.clip(intra, 0, 1)),
                "progress_mean": hack,
                "sep_scale": 10.0,
            }
        )
        targets.append(hack)
    judge = fit_learned_judge(metrics, targets)
    assert judge.n == 16
    assert judge.spearman_train >= 0.8
    # Predicted ranking should track targets.
    preds = [judge.score_metrics(m) for m in metrics]
    order_p = np.argsort(preds)
    order_t = np.argsort(targets)
    assert int(np.sum(order_p == order_t)) >= 10


def test_redesign_weights_sum_positive():
    w = redesigned_composite_weights()
    assert abs(sum(w.values()) - 1.0) < 1e-9
    c = calibrated_composite(
        offline_rank=0.5,
        separation=0.5,
        anti_hack=1.0,
        intra_scenario=0.0,
        **w,
    )
    assert 0.0 <= c <= 1.0


def test_gate_score_bounded():
    metrics = [
        {
            "offline_rank": 0.9,
            "separation": 0.9,
            "anti_hack": float(i) / 10.0,
            "intra_scenario": 0.5,
            "progress_mean": float(i),
            "sep_scale": 10.0,
        }
        for i in range(12)
    ]
    targets = [float(i) for i in range(12)]
    judge = fit_learned_judge(metrics, targets)
    for m in metrics:
        g = judge.gate_score(m)
        assert 0.0 < g < 1.0


def test_dataset_gate_args_learned_judge():
    from argparse import Namespace

    from card_plus.dataset.gate import dataset_gate_from_args

    cfg = dataset_gate_from_args(Namespace(dataset_learned_judge="results/learned_judge_g2.json"))
    assert cfg.dataset_learned_judge.endswith("learned_judge_g2.json")
