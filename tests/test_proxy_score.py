"""Env-grounded proxy scoring for zero-success short ranking."""

from __future__ import annotations

import numpy as np
import pytest

from card_plus.archive import extract_proxy_curve, extract_success_curve, score_from_curve
from card_plus.progress import (
    crowdnav_goal_progress_from_dists,
    crowdnav_soft_near_goal,
    episode_proxy_score,
)


def test_success_maps_to_one():
    assert episode_proxy_score(success=1.0, progress=0.1, collided=1.0, soft_near=0.0) == 1.0


def test_zero_success_ranks_by_progress_then_survive():
    better = episode_proxy_score(success=0.0, progress=0.8, collided=0.0, soft_near=1.0)
    worse = episode_proxy_score(success=0.0, progress=0.1, collided=1.0, soft_near=0.0)
    assert better > worse
    assert 0.0 <= worse < better <= 1.0


def test_soft_near_and_progress_helpers():
    assert crowdnav_goal_progress_from_dists(4.0, 0.4, 0.4) > 0.8
    assert crowdnav_soft_near_goal(0.5, radius=0.3) == 1.0
    assert crowdnav_soft_near_goal(3.0, radius=0.3) == 0.0


def test_extract_proxy_curve_prefers_proxy_matrix():
    lr = {
        "successes": np.zeros((2, 4), dtype=np.float64),
        "proxy_scores": np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]),
    }
    proxy = extract_proxy_curve(lr)
    success = extract_success_curve(lr)
    assert success == [0.0, 0.0]
    assert proxy[0] == pytest.approx(0.25)
    assert score_from_curve(proxy, last_k=1) == pytest.approx(0.65)


def test_extract_proxy_falls_back_to_success():
    lr = {"successes": np.array([[1.0, 0.0, 1.0, 0.0]])}
    assert extract_proxy_curve(lr) == [0.5]
