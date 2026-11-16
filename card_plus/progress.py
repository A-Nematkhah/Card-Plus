"""Task-progress measures from observations / env info only (never GT reward)."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

ProgressFn = Callable[[Dict[str, Any]], float]


def _as_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _object_and_goal(traj: Dict[str, Any]) -> tuple:
    """Return (obj_positions [T,3], goal [3]) from a stored Meta-World trajectory."""
    if "self.obj1.position" in traj:
        obj = np.asarray(traj["self.obj1.position"], dtype=np.float64)
    elif "obs" in traj:
        obs = np.asarray(traj["obs"], dtype=np.float64)
        # Verified layout: obs[4:7] = object position (current frame)
        obj = obs[:, 4:7]
    else:
        raise KeyError("Trajectory lacks object position (self.obj1.position / obs).")

    if obj.ndim == 1:
        obj = obj.reshape(1, -1)
    if obj.shape[-1] > 3:
        obj = obj[:, :3]

    if "self.goal_position" in traj:
        goal_arr = np.asarray(traj["self.goal_position"], dtype=np.float64)
        if goal_arr.ndim == 1:
            goal = goal_arr[:3]
        else:
            # use last non-zero or last row
            goal = goal_arr[-1, :3]
            if not np.any(np.abs(goal) > 1e-12) and goal_arr.shape[0] > 0:
                for i in range(goal_arr.shape[0]):
                    if np.any(np.abs(goal_arr[i, :3]) > 1e-12):
                        goal = goal_arr[i, :3]
                        break
    elif "obs" in traj:
        obs = np.asarray(traj["obs"], dtype=np.float64)
        # V2: goal is last 3 dims when fully observable; may be zeros under PO
        goal = obs[-1, -3:]
    else:
        raise KeyError("Trajectory lacks goal (self.goal_position / obs).")

    return obj, _as_float_array(goal)[:3]


def generic_object_to_goal_progress(traj: Dict[str, Any]) -> float:
    """clip(1 - min_t ||obj_t - goal|| / d_0, 0, 1)."""
    obj, goal = _object_and_goal(traj)
    dists = np.linalg.norm(obj - goal.reshape(1, 3), axis=1)
    d0 = float(dists[0])
    if d0 < 1e-12:
        # Already at goal at t=0: progress = 1 if stayed close, else 0
        return float(np.clip(1.0 - float(np.min(dists)) / 1e-6, 0.0, 1.0)) if float(np.min(dists)) > 1e-12 else 1.0
    return float(np.clip(1.0 - float(np.min(dists)) / d0, 0.0, 1.0))


def drawer_open_progress(traj: Dict[str, Any]) -> float:
    """Drawer-open: progress of object (drawer handle) toward goal along verified obs layout.

    Uses the same object→goal distance formula as the generic measure; registered
    separately so task-specific overrides remain explicit.
    """
    return generic_object_to_goal_progress(traj)


def _robot_and_goal_2d(traj: Dict[str, Any]) -> tuple:
    """Return (robot_xy [T,2], goal_xy [T,2]) from a stored CrowdNav trajectory.

    Verified CARD CrowdNav obs layout (`crowdnav_env_utils.py`):
    `obs[0:2]` robot position, `obs[4:6]` goal. Stored mapping keys
    `self.robot.position` / `self.robot.goal_position` are preferred when present.
    """
    if "self.robot.position" in traj:
        pos = np.asarray(traj["self.robot.position"], dtype=np.float64)
    elif "obs" in traj:
        obs = np.asarray(traj["obs"], dtype=np.float64)
        pos = obs[:, 0:2]
    else:
        raise KeyError("Trajectory lacks robot position (self.robot.position / obs).")

    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    pos = pos[:, :2]

    if "self.robot.goal_position" in traj:
        goal = np.asarray(traj["self.robot.goal_position"], dtype=np.float64)
    elif "obs" in traj:
        obs = np.asarray(traj["obs"], dtype=np.float64)
        goal = obs[:, 4:6]
    else:
        raise KeyError("Trajectory lacks goal (self.robot.goal_position / obs).")

    if goal.ndim == 1:
        goal = np.tile(_as_float_array(goal)[:2].reshape(1, 2), (pos.shape[0], 1))
    else:
        goal = np.asarray(goal, dtype=np.float64)[:, :2]
        if goal.shape[0] != pos.shape[0]:
            goal = np.repeat(goal[-1:, :], pos.shape[0], axis=0)
    return pos, goal


def crowdnav_goal_progress(traj: Dict[str, Any]) -> float:
    """CrowdNav progress from start distance toward the goal (never GT reward).

    Uses a blend of final distance (70%) and best approach (30%) so a traj that
    gets close then collides/timeouts is not scored like a true success.
    """
    pos, goal = _robot_and_goal_2d(traj)
    dists = np.linalg.norm(pos - goal, axis=1)
    d0 = float(dists[0])
    d_final = float(dists[-1])
    d_best = float(np.min(dists))
    if d0 < 1e-12:
        return 1.0 if d_final <= 1e-12 else 0.0
    prog_final = 1.0 - d_final / d0
    prog_best = 1.0 - d_best / d0
    return float(np.clip(0.7 * prog_final + 0.3 * prog_best, 0.0, 1.0))


PROGRESS_FNS: Dict[str, ProgressFn] = {
    "generic": generic_object_to_goal_progress,
    "drawer-open-v2": drawer_open_progress,
    "drawer-open": drawer_open_progress,
    "circle-crossing-v0": crowdnav_goal_progress,
    "circle-crossing": crowdnav_goal_progress,
    "crowdnav": crowdnav_goal_progress,
}


def get_progress_fn(task_name: Optional[str] = None) -> ProgressFn:
    if task_name is None:
        return PROGRESS_FNS["generic"]
    key = task_name.strip()
    if key in PROGRESS_FNS:
        return PROGRESS_FNS[key]
    # strip common suffixes / aliases
    for k, fn in PROGRESS_FNS.items():
        if key.startswith(k) or k.startswith(key):
            return fn
    return PROGRESS_FNS["generic"]


def compute_progress_values(
    trajs: Sequence[Dict[str, Any]],
    task_name: Optional[str] = None,
    progress_fn: Optional[ProgressFn] = None,
) -> np.ndarray:
    fn = progress_fn or get_progress_fn(task_name)
    vals = [float(fn(t)) for t in trajs]
    return np.asarray(vals, dtype=np.float64)


def traj_success_flag(traj: Dict[str, Any]) -> bool:
    infos = traj.get("infos")
    if infos is not None and len(infos) > 0:
        last = infos[-1]
        if isinstance(last, dict):
            if "success" in last:
                return bool(last["success"])
            if "is_success" in last:
                return bool(last["is_success"])
    if "success" in traj:
        return bool(traj["success"])
    return False


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann–Whitney AUC of scores vs binary labels (numpy-only)."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # AUC = P(score_pos > score_neg) + 0.5 P(equal)
    # Efficient: rank all
    order = np.argsort(scores)
    ranks = np.empty_like(scores, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # average ranks for ties
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j - 1]])
            ranks[order[i:j]] = avg
        i = j
    sum_ranks_pos = float(np.sum(ranks[labels > 0.5]))
    n_pos = float(len(pos))
    n_neg = float(len(neg))
    # Mann–Whitney U / (n_pos * n_neg)
    u = sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0
    return float(u / (n_pos * n_neg))


def validate_progress_fn(
    trajs: Sequence[Dict[str, Any]],
    task_name: Optional[str] = None,
    progress_fn: Optional[ProgressFn] = None,
    auc_min: float = 0.9,
) -> float:
    """Compute AUC of progress vs success; warn if below auc_min. Returns AUC."""
    if len(trajs) == 0:
        logger.warning("validate_progress_fn: empty trajectory pool.")
        return float("nan")
    progress = compute_progress_values(trajs, task_name=task_name, progress_fn=progress_fn)
    labels = np.asarray([1.0 if traj_success_flag(t) else 0.0 for t in trajs], dtype=np.float64)
    auc = _roc_auc(progress, labels)
    if np.isnan(auc):
        logger.warning(
            "Progress validation AUC is undefined (need both successful and failed trajectories)."
        )
    elif auc < auc_min:
        logger.warning(
            "Progress measure may be wrong for this task: AUC=%.3f < min=%.3f. Continuing anyway.",
            auc,
            auc_min,
        )
    else:
        logger.info("Progress validation AUC=%.3f (ok).", auc)
    return auc
