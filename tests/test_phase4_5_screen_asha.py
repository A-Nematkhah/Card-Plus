"""Phase 4–5 unit tests: screen rank, ASHA, hypothesis, failure, llm cache."""

from __future__ import annotations

import numpy as np

from card_plus.archive import RewardArchive, VersionRecord
from card_plus.config import CardPlusConfig
from card_plus.dataset.schema import AgentState, Episode, EpisodeLabels, ScenarioSpec, StepRecord
from card_plus.dataset.screen_rank import all_screen_metrics, combined_screen, rank_predictors
from card_plus.dataset.successive_halving import (
    SHCandidate,
    aggregate_seed_scores,
    asha_rung,
    apply_seed_scores,
)
from card_plus.failure_feedback import cluster_failure_tags, format_failure_feedback
from card_plus.hypothesis_loop import (
    HypothesisRecord,
    append_hypothesis,
    parse_hypothesis_block,
    resolve_pending,
)
from card_plus.llm_cache import get_cached_response, put_cached_response


def test_screen_metrics_and_rank():
    curve = [0.1, 0.2, 0.4, 0.5]
    m = all_screen_metrics(curve, [0.2, 0.3, 0.45, 0.55])
    assert m["success_at_1"] == 0.5
    assert combined_screen(curve) == combined_screen(curve)
    rows = [
        {"success_at_1": 0.2, "combined": 0.3, "final_success": 0.4},
        {"success_at_1": 0.5, "combined": 0.6, "final_success": 0.7},
        {"success_at_1": 0.8, "combined": 0.9, "final_success": 0.95},
    ]
    ranking = rank_predictors(rows, ["success_at_1", "combined"])
    assert ranking[0]["spearman"] >= 0.9


def test_asha_median_seeds():
    med, std, n = aggregate_seed_scores([0.2, 0.8])
    assert n == 2 and abs(med - 0.5) < 1e-9 and std > 0
    cands = [
        SHCandidate(0, 0, "a", success_rate=0.0, composite=0.9),
        SHCandidate(1, 0, "b", success_rate=0.0, composite=0.5),
    ]

    def train_fn(c: SHCandidate, seed: int) -> float:
        return 0.9 if c.slot == 0 else 0.1

    kept = asha_rung(cands, seeds=[0, 1], train_fn=train_fn)
    assert len(kept) == 1
    assert kept[0].slot == 0
    assert kept[0].n_seeds == 2


def test_rollback_uses_std_k(tmp_path):
    cfg = CardPlusConfig(use_archive_rollback=True, rollback_margin=0.05, rollback_std_k=2.0)
    arch = RewardArchive(run_dir=str(tmp_path), cfg=cfg)
    arch.add(
        VersionRecord(
            iteration=1,
            reward_code="a",
            rl_ran=True,
            version_score=0.80,
            score_std=0.10,
            n_seeds=2,
            success_curve=[0.8],
        )
    )
    # Drop of 0.08 < margin 0.05 + 2*0.10=0.25 → no rollback
    assert arch.should_rollback_against(0.72, exclude_iteration=2) is False
    # Drop of 0.30 > 0.25 → rollback
    assert arch.should_rollback_against(0.50, exclude_iteration=2) is True


def test_hypothesis_loop(tmp_path):
    parsed = parse_hypothesis_block("Hypothesis: boost progress\nExpectedEffect: +success\n")
    assert "progress" in parsed["hypothesis"]
    append_hypothesis(
        tmp_path,
        HypothesisRecord(
            iteration=1,
            hypothesis="boost progress",
            expected_effect="+success",
            metric_before=0.4,
        ),
    )
    resolved = resolve_pending(tmp_path, metric_after=0.5, min_delta=0.02)
    assert resolved is not None and resolved.verdict == "confirmed"


def _ep(outcome: str, tag_progress: bool = False) -> Episode:
    robot = AgentState(0, 0.0, 0.0, 0.0, 0.0, 0.3, 5.0, 0.0, True)
    step = StepRecord(
        t=0,
        action=np.zeros(2),
        obs=np.zeros(116),
        reward_raw=0.0,
        done=True,
        success=outcome == "success",
        collided=outcome == "collided",
        timeout=outcome == "timeout",
        robot=robot,
        humans=[],
        min_separation=1.0,
        potential=0.0,
    )
    labels = EpisodeLabels(
        outcome=outcome,
        failure_tags=["freezing"] if outcome == "timeout" else (["collision"] if outcome == "collided" else []),
        stationary_ratio=0.9 if outcome == "timeout" else 0.0,
    )
    return Episode(
        schema_version=1,
        scenario=ScenarioSpec(
            scenario_id=f"s_{outcome}",
            task_name="t",
            this_seed=0,
            phase="test",
            test_case=0,
            human_num=5,
            time_step=0.25,
            time_limit=25.0,
            randomization="none",
            difficulty={},
            initial_robot=robot,
        ),
        policy_name="p",
        steps=[step],
        labels=labels,
    )


def test_failure_clusters():
    eps = [_ep("timeout"), _ep("collided"), _ep("success")]
    counts = cluster_failure_tags(eps)
    assert counts.get("freezing") == 1 or counts.get("timeout") == 1
    text = format_failure_feedback(eps)
    assert "Failure mode" in text


def test_llm_cache(tmp_path):
    put_cached_response(tmp_path, "hello", "world", model="m")
    assert get_cached_response(tmp_path, "hello", model="m") == "world"
    assert (tmp_path / "llm_prompt_response.jsonl").is_file()
