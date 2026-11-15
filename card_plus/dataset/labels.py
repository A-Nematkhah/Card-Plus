"""Episode-level labels from recorded CrowdNav trajectories."""

from __future__ import annotations

from typing import List

import numpy as np

from card_plus.dataset.schema import Episode, EpisodeLabels
from card_plus.progress import crowdnav_goal_progress


def _outcome(episode: Episode) -> str:
    if not episode.steps:
        return "empty"
    last = episode.steps[-1]
    if last.success:
        return "success"
    if last.collided:
        return "collided"
    if last.timeout:
        return "timeout"
    if last.done:
        return "truncated"
    return "incomplete"


def _path_length(episode: Episode) -> float:
    if not episode.steps:
        return 0.0
    init = episode.scenario.initial_robot
    pts = []
    if init is not None:
        pts.append((init.px, init.py))
    for s in episode.steps:
        pts.append((s.robot.px, s.robot.py))
    total = 0.0
    for i in range(1, len(pts)):
        total += float(np.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    return total


def _stationary_ratio(episode: Episode, speed_eps: float = 0.05) -> float:
    if not episode.steps:
        return 0.0
    speeds = [float(np.hypot(s.robot.vx, s.robot.vy)) for s in episode.steps]
    return float(np.mean(np.asarray(speeds) < speed_eps))


def _failure_tags(episode: Episode, outcome: str, stationary_ratio: float) -> List[str]:
    tags: List[str] = []
    if outcome == "collided":
        tags.append("collision")
    if outcome == "timeout":
        tags.append("timeout")
    if stationary_ratio >= 0.5 and outcome != "success":
        tags.append("freezing")
    if episode.steps:
        init = episode.scenario.initial_robot
        if init is not None:
            d0 = float(np.hypot(init.px - init.gx, init.py - init.gy))
            dT = float(
                np.hypot(
                    episode.steps[-1].robot.px - episode.steps[-1].robot.gx,
                    episode.steps[-1].robot.py - episode.steps[-1].robot.gy,
                )
            )
            if d0 > 1e-6 and dT > 0.9 * d0 and outcome != "success":
                tags.append("no_progress")
    return tags


def _progress_from_episode(episode: Episode) -> float:
    """Reuse card_plus.progress on obs arrays (never GT reward)."""
    if not episode.steps:
        return 0.0
    obs = episode.obs_array()
    traj = {
        "obs": obs,
        "self.robot.position": obs[:, 0:2],
        "self.robot.goal_position": obs[:, 4:6],
    }
    return float(crowdnav_goal_progress(traj))


def label_episode(
    episode: Episode,
    *,
    discomfort_dist: float = 0.25,
    stationary_speed_eps: float = 0.05,
) -> EpisodeLabels:
    outcome = _outcome(episode)
    time_step = float(episode.extras.get("time_step", episode.scenario.time_step))
    nav_time = float(len(episode.steps) * time_step)
    path_len = _path_length(episode)
    init = episode.scenario.initial_robot
    if init is None or path_len < 1e-12:
        efficiency = 0.0
    else:
        straight = float(np.hypot(init.px - init.gx, init.py - init.gy))
        efficiency = float(straight / path_len) if path_len > 1e-12 else 0.0
    seps = [float(s.min_separation) for s in episode.steps]
    min_sep = float(min(seps)) if seps else float("inf")
    discomfort_steps = int(sum(1 for s in seps if s < discomfort_dist))
    stationary = _stationary_ratio(episode, speed_eps=stationary_speed_eps)
    progress = _progress_from_episode(episode)
    tags = _failure_tags(episode, outcome, stationary)
    return EpisodeLabels(
        outcome=outcome,
        nav_time=nav_time,
        path_length=path_len,
        path_efficiency=efficiency,
        min_sep=min_sep,
        discomfort_steps=discomfort_steps,
        stationary_ratio=stationary,
        progress=progress,
        failure_tags=tags,
    )


def attach_labels(episode: Episode, **kwargs) -> Episode:
    episode.labels = label_episode(episode, **kwargs)
    return episode
