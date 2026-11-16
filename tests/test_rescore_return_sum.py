"""Episode ranking uses cumulative (sum) return; pair ties count as 0.5."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from card_plus.dataset.rescore import (
    _episode_return,
    _pair_accuracy,
    outcome_return_auc,
)


def test_episode_return_is_sum_not_mean():
    assert _episode_return(np.array([1.0, 2.0, 3.0])) == 6.0
    assert _episode_return(np.array([])) == 0.0


def test_pair_accuracy_ties_are_half():
    eps = [
        SimpleNamespace(labels=SimpleNamespace(outcome="success")),
        SimpleNamespace(labels=SimpleNamespace(outcome="collided")),
    ]
    returns = [2.0, 2.0]
    pa = _pair_accuracy(eps, returns, "success", "collided")
    assert pa.n_pairs == 1
    assert abs(pa.accuracy - 0.5) < 1e-12


def test_outcome_auc_ties_half():
    eps = [
        SimpleNamespace(labels=SimpleNamespace(outcome="success")),
        SimpleNamespace(labels=SimpleNamespace(outcome="timeout")),
    ]
    assert abs(outcome_return_auc(eps, [2.0, 2.0]) - 0.5) < 1e-12
