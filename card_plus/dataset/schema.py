"""Versioned episode / step schema for CrowdNav offline datasets.

Pure data + save/load. No simulator imports (keeps golden / unit tests light).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# Episode pickle schema. Bump when on-disk Episode fields change incompatibly.
SCHEMA_VERSION = 1
# Dataset package format (index/split/meta). v2 adds scenario_split + cautious policy recipe.
DATASET_FORMAT_VERSION = 2

PathLike = Union[str, Path]


@dataclass
class AgentState:
    """One agent at one time (absolute world frame)."""

    agent_id: int
    px: float
    py: float
    vx: float
    vy: float
    radius: float
    gx: float
    gy: float
    is_robot: bool = False

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy],
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, agent_id: int, arr: Sequence[float], *, is_robot: bool = False) -> "AgentState":
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
        return cls(
            agent_id=int(agent_id),
            px=float(a[0]),
            py=float(a[1]),
            vx=float(a[2]),
            vy=float(a[3]),
            radius=float(a[4]),
            gx=float(a[5]),
            gy=float(a[6]),
            is_robot=bool(is_robot),
        )


@dataclass
class ScenarioSpec:
    """Fixed scenario identity for phase='test' case locking."""

    scenario_id: str
    task_name: str
    this_seed: int
    phase: str
    test_case: int
    human_num: int
    time_step: float
    time_limit: float
    randomization: str
    difficulty: Dict[str, float] = field(default_factory=dict)
    initial_robot: Optional[AgentState] = None
    initial_humans: List[AgentState] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ScenarioSpec":
        data = dict(d)
        ir = data.get("initial_robot")
        if ir is not None and not isinstance(ir, AgentState):
            data["initial_robot"] = AgentState(**ir)
        humans = data.get("initial_humans") or []
        data["initial_humans"] = [
            h if isinstance(h, AgentState) else AgentState(**h) for h in humans
        ]
        return cls(**data)


@dataclass
class StepRecord:
    """One transition after taking ``action`` (post-step state).

    ``obs`` is the flat CARD observation used by LLM rewards (post-step).
    ``reward_raw`` is the simulator GT reward from the inner CrowdNavEnv
    (before ContinuousTaskWrapper overwrite).
    """

    t: int
    action: np.ndarray
    obs: np.ndarray
    reward_raw: float
    done: bool
    success: bool
    collided: bool
    timeout: bool
    robot: AgentState
    humans: List[AgentState]
    min_separation: float
    potential: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": int(self.t),
            "action": np.asarray(self.action, dtype=np.float64),
            "obs": np.asarray(self.obs, dtype=np.float64),
            "reward_raw": float(self.reward_raw),
            "done": bool(self.done),
            "success": bool(self.success),
            "collided": bool(self.collided),
            "timeout": bool(self.timeout),
            "robot": asdict(self.robot),
            "humans": [asdict(h) for h in self.humans],
            "min_separation": float(self.min_separation),
            "potential": float(self.potential),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StepRecord":
        return cls(
            t=int(d["t"]),
            action=np.asarray(d["action"], dtype=np.float64),
            obs=np.asarray(d["obs"], dtype=np.float64),
            reward_raw=float(d["reward_raw"]),
            done=bool(d["done"]),
            success=bool(d["success"]),
            collided=bool(d["collided"]),
            timeout=bool(d["timeout"]),
            robot=AgentState(**d["robot"]),
            humans=[AgentState(**h) for h in d["humans"]],
            min_separation=float(d["min_separation"]),
            potential=float(d["potential"]),
        )


@dataclass
class EpisodeLabels:
    outcome: str = "unknown"  # success | collided | timeout | truncated
    nav_time: float = 0.0
    path_length: float = 0.0
    path_efficiency: float = 0.0
    min_sep: float = float("inf")
    discomfort_steps: int = 0
    stationary_ratio: float = 0.0
    progress: float = 0.0
    failure_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpisodeLabels":
        return cls(**dict(d))


@dataclass
class Episode:
    schema_version: int
    scenario: ScenarioSpec
    policy_name: str
    steps: List[StepRecord]
    labels: Optional[EpisodeLabels] = None
    initial_obs: Optional[np.ndarray] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return len(self.steps)

    def actions_array(self) -> np.ndarray:
        if not self.steps:
            return np.zeros((0, 2), dtype=np.float64)
        return np.stack([np.asarray(s.action, dtype=np.float64) for s in self.steps], axis=0)

    def obs_array(self) -> np.ndarray:
        if not self.steps:
            return np.zeros((0, 1), dtype=np.float64)
        return np.stack([np.asarray(s.obs, dtype=np.float64) for s in self.steps], axis=0)

    def rewards_raw_array(self) -> np.ndarray:
        return np.asarray([s.reward_raw for s in self.steps], dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "scenario": self.scenario.to_dict(),
            "policy_name": str(self.policy_name),
            "steps": [s.to_dict() for s in self.steps],
            "labels": None if self.labels is None else self.labels.to_dict(),
            "initial_obs": None
            if self.initial_obs is None
            else np.asarray(self.initial_obs, dtype=np.float64),
            "extras": dict(self.extras),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Episode":
        labels = d.get("labels")
        return cls(
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            scenario=ScenarioSpec.from_dict(d["scenario"]),
            policy_name=str(d["policy_name"]),
            steps=[StepRecord.from_dict(s) for s in d["steps"]],
            labels=None if labels is None else EpisodeLabels.from_dict(labels),
            initial_obs=None
            if d.get("initial_obs") is None
            else np.asarray(d["initial_obs"], dtype=np.float64),
            extras=dict(d.get("extras") or {}),
        )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _portable_payload(obj: Any) -> Any:
    """Replace ndarrays with nested lists so pickles work across NumPy 1.x/2.x."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _portable_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_portable_payload(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _ensure_numpy_pickle_compat() -> None:
    """Allow unpickling NumPy-2 array pickles on NumPy 1.x (numpy._core → numpy.core)."""
    import sys

    try:
        import numpy.core as _np_core
    except ImportError:
        return
    aliases = {
        "numpy._core": _np_core,
        "numpy._core.multiarray": getattr(_np_core, "multiarray", None),
        "numpy._core.numeric": getattr(_np_core, "numeric", None),
    }
    try:
        import numpy.core._multiarray_umath as _mu  # type: ignore

        aliases["numpy._core._multiarray_umath"] = _mu
    except Exception:
        aliases["numpy._core._multiarray_umath"] = aliases.get("numpy._core.multiarray")
    for name, mod in aliases.items():
        if mod is not None:
            sys.modules.setdefault(name, mod)


def save_episode(episode: Episode, path: PathLike) -> Path:
    """Save episode as ``.npz`` (pickle of portable dict) or ``.json`` (debug)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = episode.to_dict()
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(payload, default=_json_default, indent=2), encoding="utf-8")
    else:
        # Extension kept as .npz for historical paths; payload is a pickle-safe dict
        # (lists, not ndarrays) so NumPy 1.x and 2.x can both load it.
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(".npz")
        import pickle

        with path.open("wb") as f:
            pickle.dump(_portable_payload(payload), f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_episode(path: PathLike) -> Episode:
    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        import pickle

        _ensure_numpy_pickle_compat()
        with path.open("rb") as f:
            payload = pickle.load(f)
    ep = Episode.from_dict(payload)
    if ep.schema_version != SCHEMA_VERSION:
        # Forward-compatible load: still return; caller may check.
        pass
    return ep


def stack_human_array(humans: Sequence[AgentState]) -> np.ndarray:
    """[N,7] array in stable agent_id order."""
    ordered = sorted(humans, key=lambda h: h.agent_id)
    if not ordered:
        return np.zeros((0, 7), dtype=np.float64)
    return np.stack([h.as_array() for h in ordered], axis=0)


def states_close(
    a: Sequence[AgentState],
    b: Sequence[AgentState],
    *,
    atol: float = 1e-7,
) -> bool:
    return np.allclose(stack_human_array(list(a)), stack_human_array(list(b)), atol=atol)


def robot_close(a: AgentState, b: AgentState, *, atol: float = 1e-7) -> bool:
    return np.allclose(a.as_array(), b.as_array(), atol=atol)
