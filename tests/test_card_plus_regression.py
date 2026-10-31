"""Baseline regression: all CARD+ flags False → no behavioral change helpers."""

from __future__ import annotations

from card_plus.config import CardPlusConfig
from card_plus.hooks import enrich_feedback, run_optional_tpe_v2
from card_plus.tpe_v2 import should_skip_rl_for_tpe_v2, TPEv2Result


def test_defaults_all_false():
    cfg = CardPlusConfig()
    assert cfg.use_archive_rollback is False
    assert cfg.use_subreward_stats is False
    assert cfg.use_tpe_v2 is False
    assert cfg.any_enabled is False


def test_feedback_unchanged_when_flags_false():
    cfg = CardPlusConfig()
    base = "1.process feedback\n2.trajectory feedback"
    out = enrich_feedback(
        base,
        cfg,
        trajs=[],
        task_name="drawer-open-v2",
        tpe_v2=None,
        rl_skipped=False,
    )
    assert out == base


def test_tpe_v2_disabled_returns_none_and_never_skips():
    cfg = CardPlusConfig()
    assert run_optional_tpe_v2(cfg, [], pass_v1=True, acc_v1=1.0, task_name="x") is None
    fake = TPEv2Result(
        pass_v1=True, acc_v1=1.0, rho_full=1.0, rho_dense=0.0, hacked=True, verdict="hacked"
    )
    assert should_skip_rl_for_tpe_v2(cfg, fake) is False


def test_warn_mode_does_not_skip_even_if_hacked():
    cfg = CardPlusConfig(use_tpe_v2=True, tpe_v2_mode="warn")
    fake = TPEv2Result(
        pass_v1=True, acc_v1=1.0, rho_full=1.0, rho_dense=0.0, hacked=True, verdict="hacked"
    )
    assert should_skip_rl_for_tpe_v2(cfg, fake) is False
