"""Calibrated composite should not sit at ~1.0 for clearance-farm rewards."""

from card_plus.dataset.anti_hack_probes import run_anti_hack_probes
from card_plus.dataset.rescore import load_reward_fn_from_code
from card_plus.dataset.scoring import anti_hack_score, calibrated_composite, soft_gap_score
from tests.test_reward_anti_hack import FARM_CODE, PENALTY_ONLY_CODE


def test_soft_gap_equal_is_half():
    assert abs(soft_gap_score(1.0, 1.0, 1.0) - 0.5) < 1e-6


def test_anti_hack_farm_lower_than_good_reward():
    farm = anti_hack_score(run_anti_hack_probes(load_reward_fn_from_code(FARM_CODE)).metrics)
    good = anti_hack_score(run_anti_hack_probes(load_reward_fn_from_code(PENALTY_ONLY_CODE)).metrics)
    assert good > farm
    assert farm < 0.55
    assert good > 0.55


def test_calibrated_composite_penalizes_high_rank_low_hack():
    # Pairwise-perfect offline rank but weak anti-hack → not near 1.0
    c = calibrated_composite(offline_rank=0.99, separation=0.90, anti_hack=0.40)
    assert c < 0.85
    assert c > 0.5
    # Strong on all axes → higher
    c2 = calibrated_composite(offline_rank=0.99, separation=0.85, anti_hack=0.85)
    assert c2 > c
    assert c2 > 0.85
