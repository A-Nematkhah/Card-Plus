"""Synthetic trajectory probes that catch clearance-farm / flee rewards.

Offline dataset policies never deliberately flee the map, so a reward can get
composite≈1 while still paying the online agent to run away from humans.
These short synthetic rollouts close that gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from card_plus.dataset.rescore import _scalar_reward

# Match crowdnav_env_utils.OBS_DIM without importing gym-heavy utils.
_MAX_HUMANS_PAD = 20
OBS_DIM = 16 + 5 * _MAX_HUMANS_PAD

RewardFn = Callable[..., Any]


def _make_obs(
    px: float,
    py: float,
    gx: float,
    gy: float,
    *,
    min_sep: float,
    radius: float = 0.3,
    vx: float = 0.0,
    vy: float = 0.0,
) -> np.ndarray:
    obs = np.zeros(OBS_DIM, dtype=np.float64)
    obs[0] = px
    obs[1] = py
    obs[2] = vx
    obs[3] = vy
    obs[4] = gx
    obs[5] = gy
    obs[6] = radius
    obs[7] = float(min_sep)
    # Nearest-human distance ≈ separation + radii (rough).
    obs[8] = float(min_sep) + 2.0 * radius
    obs[9] = 1.0
    obs[10] = 0.0
    obs[13] = float(min_sep) + 3.0
    obs[14] = float(min_sep) + 4.0
    obs[15] = 5.0
    return obs


def _score_traj(
    reward_fn: RewardFn,
    steps: Sequence[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
) -> Tuple[float, float]:
    """Return (mean, sum) over (action, obs, pre_obs) steps."""
    set_pre = getattr(reward_fn, "set_pre_obs", None)
    reset = getattr(reward_fn, "reset_pre_obs", None)
    if reset is not None:
        reset()
    rewards: List[float] = []
    for action, obs, pre in steps:
        if set_pre is not None:
            set_pre(pre if pre is not None else obs)
        rewards.append(_scalar_reward(reward_fn(action, obs)))
    if not rewards:
        return 0.0, 0.0
    arr = np.asarray(rewards, dtype=np.float64)
    return float(arr.mean()), float(arr.sum())


def _success_steps(n: int = 12) -> List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Approach goal then terminal success (goal_dist <= radius)."""
    gx, gy = 5.0, 0.0
    out: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
    pre = None
    for i in range(n):
        # Move from x=-4 toward x=5
        t = (i + 1) / n
        px = -4.0 + t * 9.0
        py = 0.0
        # Keep modest clearance (not farming distance)
        min_sep = 0.6
        if i == n - 1:
            px, py = gx, gy  # reached
            min_sep = 0.8
        obs = _make_obs(px, py, gx, gy, min_sep=min_sep, vx=0.8, vy=0.0)
        action = np.asarray([0.8, 0.0], dtype=np.float64)
        out.append((action, obs, pre))
        pre = obs
    return out


def _progress_steps(n: int = 12) -> List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Same approach without final reach (still moving closer)."""
    gx, gy = 5.0, 0.0
    out: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
    pre = None
    for i in range(n):
        t = (i + 1) / (n + 2)  # stop short of goal
        px = -4.0 + t * 9.0
        obs = _make_obs(px, 0.0, gx, gy, min_sep=0.6, vx=0.8, vy=0.0)
        action = np.asarray([0.8, 0.0], dtype=np.float64)
        out.append((action, obs, pre))
        pre = obs
    return out


def _flee_steps(n: int = 20) -> List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Run away from goal while clearance grows (classic farm)."""
    gx, gy = 5.0, 0.0
    out: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
    pre = None
    for i in range(n):
        t = (i + 1) / n
        px = -1.0 - t * 25.0
        py = -t * 20.0
        min_sep = 0.5 + t * 30.0
        obs = _make_obs(px, py, gx, gy, min_sep=min_sep, vx=-0.9, vy=-0.7)
        action = np.asarray([-0.9, -0.7], dtype=np.float64)
        out.append((action, obs, pre))
        pre = obs
    return out


def _orbit_steps(n: int = 20) -> List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """High clearance, goal distance stays large (loiter / orbit)."""
    gx, gy = 5.0, 0.0
    out: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
    pre = None
    for i in range(n):
        ang = 2.0 * np.pi * i / n
        px = -8.0 + 1.5 * np.cos(ang)
        py = 1.5 * np.sin(ang)
        obs = _make_obs(px, py, gx, gy, min_sep=8.0 + 0.2 * np.sin(ang), vx=-0.2, vy=0.2)
        action = np.asarray([-0.2, 0.2], dtype=np.float64)
        out.append((action, obs, pre))
        pre = obs
    return out


def _early_collision_steps(n: int = 5) -> List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Short traj that ends in collision (min_sep < 0) around step 5."""
    gx, gy = 5.0, 0.0
    out: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
    pre = None
    for i in range(n):
        px = -2.0 + 0.2 * i
        # Negative separation ⇒ collided in CrowdNav convention used by rewards.
        min_sep = -0.15 if i == n - 1 else 0.4
        obs = _make_obs(px, 0.0, gx, gy, min_sep=min_sep, vx=0.5, vy=0.0)
        action = np.asarray([0.5, 0.0], dtype=np.float64)
        out.append((action, obs, pre))
        pre = obs
    return out


def _long_wander_steps(n: int = 200) -> List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Long loiter with no goal progress (suicide-vs-wander contrast)."""
    gx, gy = 5.0, 0.0
    out: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = []
    pre = None
    for i in range(n):
        # Stay far from goal; modest clearance; tiny motion.
        px = -4.0 + 0.01 * np.sin(0.1 * i)
        py = 0.2 * np.cos(0.07 * i)
        obs = _make_obs(px, py, gx, gy, min_sep=1.0, vx=0.05, vy=0.0)
        action = np.asarray([0.05, 0.0], dtype=np.float64)
        out.append((action, obs, pre))
        pre = obs
    return out


@dataclass
class AntiHackProbeResult:
    ok: bool
    reasons: List[str]
    metrics: Dict[str, float]

    def feedback_lines(self) -> List[str]:
        lines = [
            "Anti-hack probes:",
            f"  success_mean={self.metrics.get('success_mean', float('nan')):.4f}",
            f"  progress_mean={self.metrics.get('progress_mean', float('nan')):.4f}",
            f"  flee_mean={self.metrics.get('flee_mean', float('nan')):.4f}",
            f"  orbit_mean={self.metrics.get('orbit_mean', float('nan')):.4f}",
        ]
        if "suicide_mean" in self.metrics:
            lines.append(
                f"  suicide_mean={self.metrics.get('suicide_mean', float('nan')):.4f} "
                f"wander_mean={self.metrics.get('wander_mean', float('nan')):.4f}"
            )
        if self.reasons:
            lines.append("  failures: " + "; ".join(self.reasons))
        return lines


def run_suicide_probe(
    reward_fn: RewardFn,
    *,
    margin: float = 0.0,
) -> AntiHackProbeResult:
    """Early-collision episode return must be worse than long no-progress return.

    Compares **sums** (RL objective). Mean per-step can hide the hack: a tiny
    collision terminal plus large per-step stall makes crashing early look better
    in cumulative return while means stay similar.
    """
    suicide_m, suicide_s = _score_traj(reward_fn, _early_collision_steps())
    wander_m, wander_s = _score_traj(reward_fn, _long_wander_steps())
    metrics = {
        "suicide_mean": suicide_m,
        "wander_mean": wander_m,
        "suicide_sum": suicide_s,
        "wander_sum": wander_s,
        "suicide_margin": float(margin),
    }
    reasons: List[str] = []
    if not (suicide_s < wander_s - margin):
        reasons.append(
            f"suicide_sum={suicide_s:.4f} not < wander_sum={wander_s:.4f}-{margin:.2f} "
            "(early collision preferred over long no-progress wait; "
            "collision penalty should exceed horizon * per-step penalty)"
        )
    return AntiHackProbeResult(ok=len(reasons) == 0, reasons=reasons, metrics=metrics)


def run_anti_hack_probes(
    reward_fn: RewardFn,
    *,
    margin: float = 0.05,
    use_suicide_probe: bool = False,
) -> AntiHackProbeResult:
    """Require approaching-goal trajectories to beat flee/orbit hacks.

    A large success terminal can hide clearance farming if we only compare
    success vs flee means; therefore we also require progress > flee.
    """
    success_m, success_s = _score_traj(reward_fn, _success_steps())
    progress_m, progress_s = _score_traj(reward_fn, _progress_steps())
    flee_m, flee_s = _score_traj(reward_fn, _flee_steps())
    orbit_m, orbit_s = _score_traj(reward_fn, _orbit_steps())
    metrics = {
        "success_mean": success_m,
        "progress_mean": progress_m,
        "flee_mean": flee_m,
        "orbit_mean": orbit_m,
        "success_sum": success_s,
        "progress_sum": progress_s,
        "flee_sum": flee_s,
        "orbit_sum": orbit_s,
        "probe_margin": float(margin),
    }
    reasons: List[str] = []
    if not (success_m > flee_m + margin):
        reasons.append(
            f"flee_mean={flee_m:.4f} not worse than success_mean={success_m:.4f}+{margin:.2f} "
            "(reward farms clearance by fleeing)"
        )
    # Critical: terminal success must not mask dense flee farming.
    if not (progress_m > flee_m + margin):
        reasons.append(
            f"flee_mean={flee_m:.4f} not worse than progress_mean={progress_m:.4f}+{margin:.2f} "
            "(dense reward prefers fleeing over approaching the goal)"
        )
    if not (progress_s > flee_s):
        reasons.append(
            f"flee_sum={flee_s:.4f} >= progress_sum={progress_s:.4f} "
            "(cumulative flee return beats approaching the goal)"
        )
    if not (progress_m > orbit_m + margin):
        reasons.append(
            f"orbit_mean={orbit_m:.4f} not worse than progress_mean={progress_m:.4f}+{margin:.2f} "
            "(reward prefers high-clearance loitering over approaching the goal)"
        )
    if use_suicide_probe:
        suicide = run_suicide_probe(reward_fn)
        metrics.update(suicide.metrics)
        if not suicide.ok:
            reasons.extend(suicide.reasons)
    return AntiHackProbeResult(ok=len(reasons) == 0, reasons=reasons, metrics=metrics)
