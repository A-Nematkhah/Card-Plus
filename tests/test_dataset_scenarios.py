"""Scenario manifest: locked test cases + difficulty (needs CrowdNav)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from card_plus.dataset.scenarios import (
    build_manifest,
    capture_scenario,
    difficulty_descriptors,
    load_manifest,
    point_to_segment_distance,
    save_manifest,
    select_hard_scenarios,
)
from card_plus.dataset.schema import AgentState, ScenarioSpec


def test_point_to_segment_distance_midpoint():
    d = point_to_segment_distance(0.0, 1.0, -1.0, 0.0, 1.0, 0.0)
    assert d == pytest.approx(1.0)


def test_difficulty_counts_humans_near_path():
    robot = AgentState(0, 0.0, 0.0, 0.0, 0.0, 0.3, 4.0, 0.0, True)
    on_path = AgentState(0, 2.0, 0.0, 0.0, 0.0, 0.3, 2.0, 0.0, False)
    far = AgentState(1, 2.0, 5.0, 0.0, 0.0, 0.3, 2.0, 5.0, False)
    d = difficulty_descriptors(robot, [on_path, far], corridor=1.0)
    assert d["humans_near_path"] == 1.0
    assert d["human_num"] == 2.0


def test_capture_scenario_deterministic():
    a = capture_scenario(3, this_seed=42)
    b = capture_scenario(3, this_seed=42)
    assert a.scenario_id == b.scenario_id
    assert a.phase == "test"
    assert a.test_case == 3
    assert a.initial_robot is not None and b.initial_robot is not None
    assert np.allclose(a.initial_robot.as_array(), b.initial_robot.as_array())
    ha = np.stack([h.as_array() for h in a.initial_humans])
    hb = np.stack([h.as_array() for h in b.initial_humans])
    assert np.allclose(ha, hb)
    assert a.human_num == len(a.initial_humans)
    assert a.human_num >= 5


def test_different_cases_differ():
    a = capture_scenario(0, this_seed=42)
    b = capture_scenario(1, this_seed=42)
    assert a.scenario_id != b.scenario_id
    assert not np.allclose(a.initial_robot.as_array(), b.initial_robot.as_array())


def test_manifest_save_load(tmp_path: Path):
    specs = build_manifest(4, this_seed=42)
    path = save_manifest(specs, tmp_path / "manifest.json")
    loaded = load_manifest(path)
    assert len(loaded) == 4
    assert loaded[2].test_case == 2
    assert "humans_near_path" in loaded[0].difficulty


def _fake_spec(case: int, near: float, clearance: float) -> ScenarioSpec:
    return ScenarioSpec(
        scenario_id=f"case{case}",
        task_name="circle-crossing-v0",
        this_seed=42,
        phase="test",
        test_case=case,
        human_num=20,
        time_step=0.25,
        time_limit=50.0,
        randomization="no_random",
        difficulty={
            "humans_near_path": float(near),
            "min_clearance_to_path": float(clearance),
        },
    )


def test_select_hard_scenarios_prefers_near_path():
    pool = [
        _fake_spec(0, near=0, clearance=2.0),
        _fake_spec(1, near=3, clearance=0.5),
        _fake_spec(2, near=2, clearance=0.2),
        _fake_spec(3, near=1, clearance=1.0),
    ]
    picked = select_hard_scenarios(pool, keep=2, min_near_path=2.0)
    assert [s.test_case for s in picked] == [1, 2]


def test_select_hard_strict_does_not_soft_fill():
    pool = [_fake_spec(i, near=0, clearance=float(i)) for i in range(5)]
    picked = select_hard_scenarios(pool, keep=3, min_near_path=9.0)
    assert picked == []


def test_select_hard_soft_fill_when_enabled():
    pool = [_fake_spec(i, near=0, clearance=float(i)) for i in range(5)]
    picked = select_hard_scenarios(pool, keep=3, min_near_path=9.0, allow_soft_fill=True)
    assert len(picked) == 3
    assert picked[0].test_case == 0


def test_select_min_clearance_rejects_path_intersect():
    pool = [
        _fake_spec(0, near=2, clearance=-0.2),
        _fake_spec(1, near=2, clearance=0.5),
        _fake_spec(2, near=3, clearance=0.1),
    ]
    picked = select_hard_scenarios(pool, keep=3, min_near_path=0.0, min_clearance=0.2)
    assert [s.test_case for s in picked] == [1]


def test_select_balanced_mixes_buckets():
    pool = (
        [_fake_spec(i, near=0, clearance=1.0) for i in range(10)]
        + [_fake_spec(100 + i, near=1, clearance=0.35) for i in range(10)]
        + [_fake_spec(200 + i, near=3, clearance=0.3) for i in range(10)]
    )
    picked = select_hard_scenarios(
        pool, keep=10, min_near_path=0.0, min_clearance=0.2, selection="balanced"
    )
    assert len(picked) == 10
    from card_plus.dataset.scenarios import _bucket_name

    counts = Counter(_bucket_name(s) for s in picked)
    assert counts["easy"] >= 4
    assert counts["hard"] >= 1
