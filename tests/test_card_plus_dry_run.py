"""Dry-run loop smoke test."""

from __future__ import annotations

from card_plus.config import CardPlusConfig
from card_plus.dry_run import run_dry_loop


def test_dry_run_two_iterations(tmp_path):
    cfg = CardPlusConfig(
        use_archive_rollback=True,
        use_subreward_stats=True,
        use_tpe_v2=True,
        tpe_v2_mode="warn",
    )
    result = run_dry_loop(tmp_path, cfg, n_iters=2)
    assert result["llm_calls"] == 2
    assert len(result["iterations"]) == 2
    assert (tmp_path / "card_plus_log.jsonl").exists()
    assert result["best_code"] is not None
