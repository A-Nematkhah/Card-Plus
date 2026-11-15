"""Rollout recorder: capture per-step absolute state from ``env.sim``.

Reads reward from the inner ``CrowdNavEnv`` (before ContinuousTaskWrapper), so
``reward_raw`` is true simulator GT even if a wrapper later overwrites
``info['ground_truth_reward']``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from card_plus.dataset.policies import PolicyFn, get_policy
from card_plus.dataset.scenarios import (
    DEFAULT_ENV_CONFIG,
    DEFAULT_PHASE,
    DEFAULT_TASK,
    DEFAULT_THIS_SEED,
    make_dataset_env,
    snapshot_agents,
)
from card_plus.dataset.schema import (
    SCHEMA_VERSION,
    AgentState,
    Episode,
    ScenarioSpec,
    StepRecord,
)

# Reward constants matching CrowdNav build_config / calc_reward (holonomic).
SUCCESS_REWARD = 10.0
COLLISION_PENALTY = -20.0
DISCOMFORT_DIST = 0.25
DISCOMFORT_PENALTY_FACTOR = 10.0
POT_FACTOR = 2.0
TIME_STEP_DEFAULT = 0.25


def min_separation(robot: AgentState, humans: Sequence[AgentState]) -> float:
    best = float("inf")
    for h in humans:
        d = float(np.hypot(robot.px - h.px, robot.py - h.py) - robot.radius - h.radius)
        if d < best:
            best = d
    return 15.0 if best == float("inf") else best


def recompute_gt_reward(
    robot_at_calc: AgentState,
    humans_at_calc: Sequence[AgentState],
    *,
    success: bool,
    collided: bool,
    timeout: bool,
    potential: float,
    time_step: float = TIME_STEP_DEFAULT,
    success_reward: float = SUCCESS_REWARD,
    collision_penalty: float = COLLISION_PENALTY,
    discomfort_dist: float = DISCOMFORT_DIST,
    discomfort_penalty_factor: float = DISCOMFORT_PENALTY_FACTOR,
    pot_factor: float = POT_FACTOR,
) -> tuple:
    """Offline GT matching CrowdNav holonomic ``calc_reward`` (circle danger).

    Important: ``CrowdSimVarNum.step`` calls ``calc_reward`` *before* integrating
    the action, so ``robot_at_calc`` / ``humans_at_calc`` are the pre-step states
    (initial, or the previous step's post-step snapshot).

    ``potential`` is the simulator's running value (``-goal_dist``), updated only
    on the progress branch — not on discomfort / terminal steps.

    Returns ``(reward, potential_after)``.
    """
    if timeout and not success and not collided:
        return 0.0, float(potential)
    if collided:
        return float(collision_penalty), float(potential)
    if success:
        return float(success_reward), float(potential)
    dmin = min_separation(robot_at_calc, humans_at_calc)
    if dmin < discomfort_dist:
        r = float((dmin - discomfort_dist) * discomfort_penalty_factor * time_step)
        return r, float(potential)
    d_cur = float(
        np.hypot(robot_at_calc.px - robot_at_calc.gx, robot_at_calc.py - robot_at_calc.gy)
    )
    r = float(pot_factor * (-d_cur - float(potential)))
    return r, float(-d_cur)


def step_circle_gt(env, action: np.ndarray):
    """Step with test-case layout but CARD-train circle GT (not future-danger).

    Scenarios stay locked via ``phase='test'`` + ``test_case``. Upstream
    ``CrowdSimVarNum.step`` uses future-danger when ``phase=='test'``; CARD's
    default env keeps ``phase='train'`` (circle). Dataset recording matches CARD
    GT by flipping phase only for the duration of ``env.step``.
    """
    sim = env.sim
    locked_phase = sim.phase
    sim.phase = "train"
    try:
        return env.step(action)
    finally:
        sim.phase = locked_phase



def capture_step_state(sim) -> tuple:
    robot, humans = snapshot_agents(sim)
    dmin = min_separation(robot, humans)
    potential = -float(np.hypot(robot.px - robot.gx, robot.py - robot.gy))
    return robot, humans, dmin, potential


def record_episode(
    scenario: ScenarioSpec,
    policy: PolicyFn,
    *,
    policy_name: str = "policy",
    max_steps: Optional[int] = None,
    env_config: Optional[Dict[str, Any]] = None,
) -> Episode:
    """Roll out one locked scenario; store post-step absolute state + raw GT."""
    cfg = dict(DEFAULT_ENV_CONFIG)
    if env_config:
        cfg.update(env_config)
    # Prefer scenario randomization if present.
    if scenario.randomization:
        cfg.setdefault("randomization", scenario.randomization)
        cfg["randomization"] = scenario.randomization

    env = make_dataset_env(
        scenario.task_name,
        this_seed=scenario.this_seed,
        phase=scenario.phase,
        test_case=scenario.test_case,
        env_config=cfg,
    )
    try:
        obs = env.reset()
        initial_obs = np.asarray(obs, dtype=np.float64).copy()
        init_robot, init_humans, _, _ = capture_step_state(env.sim)
        # Bind initials to the recording reset (same lock as scenario id).
        scenario = ScenarioSpec(
            scenario_id=scenario.scenario_id,
            task_name=scenario.task_name,
            this_seed=scenario.this_seed,
            phase=scenario.phase,
            test_case=scenario.test_case,
            human_num=int(len(init_humans)),
            time_step=float(scenario.time_step),
            time_limit=float(scenario.time_limit),
            randomization=scenario.randomization,
            difficulty=dict(scenario.difficulty),
            initial_robot=init_robot,
            initial_humans=init_humans,
            extra=dict(scenario.extra),
        )
        horizon = int(max_steps) if max_steps is not None else int(env.max_episode_steps)
        steps: List[StepRecord] = []
        time_step = float(scenario.time_step)

        for t in range(horizon):
            action = np.asarray(policy(obs), dtype=np.float32).reshape(2)
            # Inner CrowdNavEnv only (no ContinuousTaskWrapper). Circle GT.
            obs, reward, done, info = step_circle_gt(env, action)
            robot, humans, dmin, potential = capture_step_state(env.sim)
            success = bool(info.get("success", 0))
            collided = bool(info.get("collided", 0))
            timeout = bool(info.get("timeout", 0))
            steps.append(
                StepRecord(
                    t=t,
                    action=action.astype(np.float64),
                    obs=np.asarray(obs, dtype=np.float64).copy(),
                    reward_raw=float(reward),
                    done=bool(done),
                    success=success,
                    collided=collided,
                    timeout=timeout,
                    robot=robot,
                    humans=humans,
                    min_separation=float(dmin),
                    potential=float(potential),
                )
            )
            if done:
                break

        return Episode(
            schema_version=SCHEMA_VERSION,
            scenario=scenario,
            policy_name=str(policy_name),
            steps=steps,
            labels=None,
            initial_obs=initial_obs,
            extras={
                "time_step": time_step,
                # Scenarios use phase=test locks; GT uses circle danger (CARD train).
                "reward_mode": "circle_gt",
            },
        )
    finally:
        env.close()


def record_episode_by_case(
    test_case: int,
    policy: PolicyFn,
    *,
    policy_name: str = "policy",
    task_name: str = DEFAULT_TASK,
    this_seed: int = DEFAULT_THIS_SEED,
    phase: str = DEFAULT_PHASE,
    env_config: Optional[Dict[str, Any]] = None,
    max_steps: Optional[int] = None,
) -> Episode:
    from card_plus.dataset.scenarios import capture_scenario

    scenario = capture_scenario(
        test_case,
        task_name=task_name,
        this_seed=this_seed,
        phase=phase,
        env_config=env_config,
    )
    return record_episode(
        scenario,
        policy,
        policy_name=policy_name,
        max_steps=max_steps,
        env_config=env_config,
    )


def replay_actions(
    scenario: ScenarioSpec,
    actions: Sequence[np.ndarray],
    *,
    env_config: Optional[Dict[str, Any]] = None,
) -> Episode:
    """Re-simulate a fixed action sequence on the locked scenario."""
    actions = [np.asarray(a, dtype=np.float32).reshape(2) for a in actions]
    idx = {"i": 0}

    def _policy(obs: np.ndarray) -> np.ndarray:
        del obs
        i = idx["i"]
        if i >= len(actions):
            return np.zeros(2, dtype=np.float32)
        a = actions[i]
        idx["i"] = i + 1
        return a

    return record_episode(
        scenario,
        _policy,
        policy_name="replay",
        max_steps=len(actions),
        env_config=env_config,
    )


def gt_rewards_from_episode(episode: Episode) -> np.ndarray:
    """Recompute GT from stored pre-step states (circle danger, running potential)."""
    if not episode.steps:
        return np.zeros(0, dtype=np.float64)
    time_step = float(episode.extras.get("time_step", episode.scenario.time_step))
    init = episode.scenario.initial_robot
    init_humans = episode.scenario.initial_humans
    if init is None:
        raise ValueError("Episode scenario lacks initial_robot for GT recompute.")
    potential = -float(np.hypot(init.px - init.gx, init.py - init.gy))
    prev_robot, prev_humans = init, list(init_humans)
    out = []
    for step in episode.steps:
        # calc_reward runs before the action; flags on this step refer to that check.
        reward, potential = recompute_gt_reward(
            prev_robot,
            prev_humans,
            success=step.success,
            collided=step.collided,
            timeout=step.timeout,
            potential=potential,
            time_step=time_step,
        )
        out.append(reward)
        prev_robot, prev_humans = step.robot, list(step.humans)
    return np.asarray(out, dtype=np.float64)


def episodes_states_equal(a: Episode, b: Episode, *, atol: float = 1e-6) -> bool:
    if a.T != b.T:
        return False
    if a.initial_obs is not None and b.initial_obs is not None:
        if not np.allclose(a.initial_obs, b.initial_obs, atol=atol):
            return False
    for sa, sb in zip(a.steps, b.steps):
        if not np.allclose(sa.action, sb.action, atol=atol):
            return False
        if not np.allclose(sa.obs, sb.obs, atol=atol):
            return False
        if not np.allclose(sa.robot.as_array(), sb.robot.as_array(), atol=atol):
            return False
        ha = np.stack([h.as_array() for h in sorted(sa.humans, key=lambda x: x.agent_id)])
        hb = np.stack([h.as_array() for h in sorted(sb.humans, key=lambda x: x.agent_id)])
        if not np.allclose(ha, hb, atol=atol):
            return False
        if abs(sa.reward_raw - sb.reward_raw) > atol:
            return False
    return True


def record_with_named_policy(
    scenario: ScenarioSpec,
    policy_name: str,
    **policy_kwargs: Any,
) -> Episode:
    policy = get_policy(policy_name, **policy_kwargs)
    return record_episode(scenario, policy, policy_name=policy_name)
