"""Collect policy expansion + noisy wrapper (no simulator)."""

from __future__ import annotations

import numpy as np

from card_plus.dataset.collect import _expand_policy_jobs
from card_plus.dataset.policies import get_policy, wrap_noisy_policy


def test_expand_random_and_noisy_jobs():
    jobs = _expand_policy_jobs(
        ["go_to_goal", "random", "noisy_go_to_goal"],
        random_seeds=[0, 1],
        noise_seeds=[7],
        noise_std=0.3,
        sac_path=None,
    )
    names = [j[0] for j in jobs]
    assert "go_to_goal" in names
    assert "random_s0" in names and "random_s1" in names
    assert any(n.startswith("noisy_goto_") for n in names)


def test_noisy_policy_changes_action():
    base = get_policy("go_to_goal")
    noisy = wrap_noisy_policy(base, noise_std=0.5, seed=0)
    obs = np.zeros(16 + 5 * 20, dtype=np.float64)
    obs[0:2] = (0.0, 0.0)
    obs[4:6] = (1.0, 0.0)
    a0 = base(obs)
    a1 = noisy(obs)
    assert a0.shape == (2,)
    assert a1.shape == (2,)
    assert not np.allclose(a0, a1)
