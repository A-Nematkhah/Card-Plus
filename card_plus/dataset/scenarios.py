"""Fixed CrowdNav test-phase scenario manifest + difficulty descriptors.

Does not train or roll out a policy. Only resets locked ``test`` cases and
records initial conditions. Leaves CARD ``CrowdNavEnv`` defaults unchanged;
phase/test_case are set on the env instance after construction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from card_plus.dataset.schema import AgentState, ScenarioSpec

PathLike = Union[str, Path]

DEFAULT_THIS_SEED = 42
DEFAULT_TASK = "circle-crossing-v0"
DEFAULT_PHASE = "test"
# Match EvoNav/CARD density; difficulty uses geometry, not human_num_range.
DEFAULT_ENV_CONFIG: Dict[str, Any] = {
    "randomization": "no_random",
}


def point_to_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Distance from point P to segment AB."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    if denom < 1e-16:
        return math.hypot(apx, apy)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    qx, qy = ax + t * abx, ay + t * aby
    return math.hypot(px - qx, py - qy)


def difficulty_descriptors(
    robot: AgentState,
    humans: Sequence[AgentState],
    *,
    corridor: float = 1.0,
) -> Dict[str, float]:
    """Geometry-based hardness proxies (density is fixed at human_num=20).

    ``humans_near_path``: humans whose clearance to the start→goal segment is
    below ``corridor`` (meters). Better stratification key than raw human_num
    while ``human_num_range=0``.
    """
    ax, ay = float(robot.px), float(robot.py)
    bx, by = float(robot.gx), float(robot.gy)
    start_goal = math.hypot(bx - ax, by - ay)
    near = 0
    min_clear = float("inf")
    for h in humans:
        d = point_to_segment_distance(float(h.px), float(h.py), ax, ay, bx, by)
        clear = d - float(h.radius) - float(robot.radius)
        min_clear = min(min_clear, clear)
        if clear < corridor:
            near += 1
    if min_clear == float("inf"):
        min_clear = 15.0
    return {
        "start_goal_dist": float(start_goal),
        "humans_near_path": float(near),
        "min_clearance_to_path": float(min_clear),
        "human_num": float(len(humans)),
        "corridor_m": float(corridor),
    }


def snapshot_agents(sim) -> tuple:
    """Return (robot AgentState, humans with stable ids = list index at reset)."""
    robot = sim.robot
    robot_state = AgentState(
        agent_id=0,
        px=float(robot.px),
        py=float(robot.py),
        vx=float(robot.vx),
        vy=float(robot.vy),
        radius=float(robot.radius),
        gx=float(robot.gx),
        gy=float(robot.gy),
        is_robot=True,
    )
    humans: List[AgentState] = []
    for i, human in enumerate(sim.humans):
        humans.append(
            AgentState(
                agent_id=int(i),
                px=float(human.px),
                py=float(human.py),
                vx=float(human.vx),
                vy=float(human.vy),
                radius=float(human.radius),
                gx=float(human.gx),
                gy=float(human.gy),
                is_robot=False,
            )
        )
    return robot_state, humans


def make_dataset_env(
    task_name: str = DEFAULT_TASK,
    *,
    this_seed: int = DEFAULT_THIS_SEED,
    phase: str = DEFAULT_PHASE,
    test_case: int = 0,
    env_config: Optional[Dict[str, Any]] = None,
):
    """Build CrowdNavEnv locked to a single test case (baseline class untouched)."""
    from code_generation.self_reflection.crowdnav_env_utils import CrowdNavEnv

    cfg = dict(DEFAULT_ENV_CONFIG)
    if env_config:
        cfg.update(env_config)
    env = CrowdNavEnv(task_name=task_name, env_config=cfg)
    env.seed(int(this_seed))
    env.sim.phase = str(phase)
    env.sim.test_case = int(test_case)
    return env


def capture_scenario(
    test_case: int,
    *,
    task_name: str = DEFAULT_TASK,
    this_seed: int = DEFAULT_THIS_SEED,
    phase: str = DEFAULT_PHASE,
    env_config: Optional[Dict[str, Any]] = None,
    corridor: float = 1.0,
) -> ScenarioSpec:
    """Reset one locked case; record initial agents + difficulty (no policy)."""
    env = make_dataset_env(
        task_name,
        this_seed=this_seed,
        phase=phase,
        test_case=test_case,
        env_config=env_config,
    )
    try:
        env.reset()
        robot, humans = snapshot_agents(env.sim)
        diff = difficulty_descriptors(robot, humans, corridor=corridor)
        cfg = env.config
        return ScenarioSpec(
            scenario_id=f"{task_name}__seed{this_seed}__{phase}__case{test_case:04d}",
            task_name=task_name,
            this_seed=int(this_seed),
            phase=str(phase),
            test_case=int(test_case),
            human_num=int(len(humans)),
            time_step=float(cfg.env.time_step),
            time_limit=float(cfg.env.time_limit),
            randomization=str(
                (env_config or DEFAULT_ENV_CONFIG).get("randomization", "no_random")
            ),
            difficulty=diff,
            initial_robot=robot,
            initial_humans=humans,
            extra={},
        )
    finally:
        env.close()


def build_manifest(
    n_cases: int = 16,
    *,
    task_name: str = DEFAULT_TASK,
    this_seed: int = DEFAULT_THIS_SEED,
    phase: str = DEFAULT_PHASE,
    env_config: Optional[Dict[str, Any]] = None,
    corridor: float = 1.0,
    case_offset: int = 0,
) -> List[ScenarioSpec]:
    """Capture cases ``case_offset .. case_offset+n_cases-1`` (no rollouts)."""
    specs: List[ScenarioSpec] = []
    for i in range(int(n_cases)):
        specs.append(
            capture_scenario(
                case_offset + i,
                task_name=task_name,
                this_seed=this_seed,
                phase=phase,
                env_config=env_config,
                corridor=corridor,
            )
        )
    return specs


def _difficulty_sort_key(spec: ScenarioSpec) -> tuple:
    """Harder first: more humans on path, then tighter clearance."""
    d = spec.difficulty
    return (
        -float(d.get("humans_near_path", 0.0)),
        float(d.get("min_clearance_to_path", 15.0)),
        int(spec.test_case),
    )


def select_hard_scenarios(
    specs: Sequence[ScenarioSpec],
    keep: int,
    *,
    min_near_path: float = 0.0,
    max_clearance: Optional[float] = None,
) -> List[ScenarioSpec]:
    """Keep up to ``keep`` hardest scenarios passing difficulty filters.

    If filters leave fewer than ``keep``, fill from the remaining pool (still
    hardest-first) so collection does not abort on sparse hard cases.
    """
    keep = int(keep)
    if keep <= 0:
        return []
    pool = list(specs)

    def _pass(s: ScenarioSpec) -> bool:
        near = float(s.difficulty.get("humans_near_path", 0.0))
        if near < float(min_near_path):
            return False
        if max_clearance is not None:
            if float(s.difficulty.get("min_clearance_to_path", 15.0)) > float(max_clearance):
                return False
        return True

    preferred = sorted([s for s in pool if _pass(s)], key=_difficulty_sort_key)
    if len(preferred) >= keep:
        return preferred[:keep]
    preferred_ids = {s.scenario_id for s in preferred}
    filler = sorted(
        [s for s in pool if s.scenario_id not in preferred_ids],
        key=_difficulty_sort_key,
    )
    out = preferred + filler
    return out[:keep]


def build_stratified_manifest(
    keep: int,
    *,
    pool: Optional[int] = None,
    min_near_path: float = 0.0,
    max_clearance: Optional[float] = None,
    task_name: str = DEFAULT_TASK,
    this_seed: int = DEFAULT_THIS_SEED,
    phase: str = DEFAULT_PHASE,
    env_config: Optional[Dict[str, Any]] = None,
    corridor: float = 1.0,
    case_offset: int = 0,
) -> List[ScenarioSpec]:
    """Scan ``pool`` test cases, keep ``keep`` hardest under difficulty filters.

    ``pool`` defaults to ``keep`` (no oversample). Use e.g. pool=500, keep=200,
    min_near_path=2 for a harder held-out set.
    """
    keep = int(keep)
    pool_n = int(pool) if pool is not None else keep
    if pool_n < keep:
        raise ValueError(f"pool ({pool_n}) must be >= keep ({keep})")
    scanned = build_manifest(
        pool_n,
        task_name=task_name,
        this_seed=this_seed,
        phase=phase,
        env_config=env_config,
        corridor=corridor,
        case_offset=case_offset,
    )
    return select_hard_scenarios(
        scanned,
        keep,
        min_near_path=min_near_path,
        max_clearance=max_clearance,
    )


def save_manifest(specs: Sequence[ScenarioSpec], path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "n": len(specs),
        "scenarios": [s.to_dict() for s in specs],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_manifest(path: PathLike) -> List[ScenarioSpec]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ScenarioSpec.from_dict(s) for s in payload["scenarios"]]


def summarize_manifest(specs: Sequence[ScenarioSpec]) -> Dict[str, Any]:
    near = [float(s.difficulty.get("humans_near_path", 0.0)) for s in specs]
    clear = [float(s.difficulty.get("min_clearance_to_path", 0.0)) for s in specs]
    return {
        "n": len(specs),
        "humans_near_path_mean": float(np.mean(near)) if near else 0.0,
        "humans_near_path_max": float(np.max(near)) if near else 0.0,
        "min_clearance_mean": float(np.mean(clear)) if clear else 0.0,
        "scenario_ids": [s.scenario_id for s in specs],
    }
