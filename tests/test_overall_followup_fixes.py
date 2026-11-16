"""Follow-up fixes from the post §1–4 overall audit."""

from __future__ import annotations

from types import SimpleNamespace

from card_plus.dataset.scoring import calibrated_composite
from code_generation.self_reflection.crowdnav_sh_select import maybe_select_reward_via_sh
from code_generation.self_reflection.utils import trajectories_return_check


def test_sh_select_is_noop_when_flag_off():
    assert maybe_select_reward_via_sh(
        SimpleNamespace(use_successive_halving=False),
        run_dir=".",
        code_step=0,
        specific_code_path="x",
        general_code_path="y",
    ) is None


def test_sh_select_keeps_composite_winner_when_flag_on(capsys):
    out = maybe_select_reward_via_sh(
        SimpleNamespace(use_successive_halving=True),
        run_dir=".",
        code_step=0,
        specific_code_path="x",
        general_code_path="y",
    )
    assert out is None
    captured = capsys.readouterr().out
    assert "crowdnav_finalists" in captured or "per-candidate" in captured


def test_calibrated_nan_intra_does_not_inflate():
    with_nan = calibrated_composite(
        offline_rank=1.0, separation=1.0, anti_hack=1.0, intra_scenario=float("nan")
    )
    as_zero = calibrated_composite(
        offline_rank=1.0, separation=1.0, anti_hack=1.0, intra_scenario=0.0
    )
    assert abs(with_nan - as_zero) < 1e-12
    assert with_nan < 1.0


def test_maniskill_return_check_resets_between_trajs():
    class _Env:
        def __init__(self):
            self.resets = 0
            self.steps = 0

        def reset(self):
            self.resets += 1
            return None

        def step(self, action):
            self.steps += 1
            return None, float(action), False, {}

    env = _Env()

    def reward_fn(**kwargs):
        raise AssertionError("maniskill branch should use env.step, not reward_fn")

    trajs = [
        {
            "obs": [0, 0],
            "actions": [1.0, 2.0],
            "infos": [{"success": True}, {"success": True}],
        },
        {
            "obs": [0, 0],
            "actions": [3.0, 4.0],
            "infos": [{"success": False}, {"success": False}],
        },
    ]
    _c, _a, _r, _p, _fb = trajectories_return_check(
        reward_function=reward_fn,
        trajectories=trajs,
        mapping_dicts={},
        threshold=0.5,
        benchmark_name="maniskill",
        env=env,
    )
    assert env.resets == 2
    assert env.steps == 4
