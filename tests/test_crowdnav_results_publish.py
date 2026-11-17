"""Tests for curated CrowdNav results publishing."""

from pathlib import Path

from code_generation.self_reflection.crowdnav_results import (
    infer_stack_tag,
    publish_crowdnav_results,
)


def test_infer_stack_tag_from_path():
    assert "sac" in infer_stack_tag(Path("a/SAC_1/b/2026-11-16-14-34-51"))
    assert "ppo_gst" in infer_stack_tag(Path("a/PPO_GST_1/b/stamp"))


def test_publish_crowdnav_results_layout(tmp_path: Path):
    log = tmp_path / "logs" / "SAC_1" / "2026-11-16-14-34-51"
    best = log / "card_plus_best"
    best.mkdir(parents=True)
    (best / "best_model.pt").write_bytes(b"weights")
    (best / "specific.py").write_text("reward = 1\n", encoding="utf-8")
    (best / "version.txt").write_text("finalist_iter_5", encoding="utf-8")
    (best / "final_eval.json").write_text(
        '{"success_rate": 0.92, "n_episodes": 50, "eval_seed": 1001}',
        encoding="utf-8",
    )
    (best / "winner_rollout_seed42.gif").write_bytes(b"gif")
    fin = log / "finalists"
    fin.mkdir(parents=True)
    (fin / "report.json").write_text('{"best_iteration": 5}', encoding="utf-8")

    repo = tmp_path / "repo"
    out = publish_crowdnav_results(log, repo_root=repo)
    assert out.name.endswith("__sac_h10")
    assert (out / "models" / "best_model.pt").is_file()
    assert (out / "models" / "specific.py").is_file()
    assert (out / "eval" / "final_eval.json").is_file()
    assert (out / "eval" / "finalists_report.json").is_file()
    assert (out / "gifs" / "seed_42.gif").is_file()
    assert (out / "meta.json").is_file()
    assert (out / "SOURCE.txt").is_file()
