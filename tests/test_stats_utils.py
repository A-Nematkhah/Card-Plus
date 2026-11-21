"""Unit tests for card_plus.stats_utils."""

from card_plus.stats_utils import (
    bootstrap_summary,
    format_ci,
    per_seed_report,
    wilson_ci,
)


def test_wilson_ci_known_range():
    w = wilson_ci(9, 10)
    assert 0.8 < w.mean < 1.0
    assert 0.0 <= w.low <= w.mean <= w.high <= 1.0
    assert w.n == 10


def test_wilson_ci_empty():
    w = wilson_ci(0, 0)
    assert w.n == 0
    assert w.mean != w.mean  # NaN


def test_bootstrap_summary_stable():
    s = bootstrap_summary([0.2, 0.4, 0.6, 0.8], n_boot=500, seed=1)
    assert s["n"] == 4.0
    assert abs(s["mean"] - 0.5) < 1e-9
    assert s["mean_low"] <= s["mean"] <= s["mean_high"]
    assert s["median_low"] <= s["median"] <= s["median_high"]


def test_per_seed_report_bernoulli():
    rep = per_seed_report({1: [1, 1, 0], 2: [1, 0, 0]}, n_boot=200, seed=0)
    assert "1" in rep["per_seed"] and "2" in rep["per_seed"]
    pooled = rep["pooled"]
    assert "mean" in pooled and "low" in pooled


def test_format_ci():
    assert format_ci(0.92, 0.85, 0.96) == "0.920 [0.850, 0.960]"
