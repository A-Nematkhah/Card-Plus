"""Thin optional bridge: CARD+ → reward_triage (pip install -e).

All triage flags default False. When every flag is off this module must not
change CARD control flow (callers gate on ``TriageFlags.any_enabled``).
TPE v1/v2 are never replaced or skipped here.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class TriageFlags:
    triage_validate: bool = False
    triage_cache: bool = False
    triage_plateau_stop: bool = False
    triage_shadow: bool = False
    plateau_margin: float = 2.0
    plateau_patience: int = 1
    cache_root: Optional[str] = None

    @property
    def any_enabled(self) -> bool:
        return bool(
            self.triage_validate
            or self.triage_cache
            or self.triage_plateau_stop
            or self.triage_shadow
        )


def add_triage_arguments(parser: Any) -> None:
    g = parser.add_argument_group("reward_triage (optional)")
    g.add_argument("--triage-validate", action="store_true", default=False)
    g.add_argument("--triage-cache", action="store_true", default=False)
    g.add_argument("--triage-plateau-stop", action="store_true", default=False)
    g.add_argument("--triage-shadow", action="store_true", default=False)
    g.add_argument("--triage-plateau-margin", type=float, default=2.0)
    g.add_argument("--triage-plateau-patience", type=int, default=1)
    g.add_argument(
        "--triage-cache-root",
        type=str,
        default=None,
        help="Label-cache directory (default: <log_path>/triage_label_cache).",
    )


def triage_flags_from_args(args: Any) -> TriageFlags:
    return TriageFlags(
        triage_validate=bool(getattr(args, "triage_validate", False)),
        triage_cache=bool(getattr(args, "triage_cache", False)),
        triage_plateau_stop=bool(getattr(args, "triage_plateau_stop", False)),
        triage_shadow=bool(getattr(args, "triage_shadow", False)),
        plateau_margin=float(getattr(args, "triage_plateau_margin", 2.0)),
        plateau_patience=int(getattr(args, "triage_plateau_patience", 1)),
        cache_root=getattr(args, "triage_cache_root", None),
    )


def _require_triage():
    try:
        import reward_triage  # noqa: F401
        from reward_triage.label_cache import LabelCache, make_cache_key, normalize_code
        from reward_triage.plateau import PlateauStop
        from reward_triage.hosts.card_triage import run_tier0_on_code
        from reward_triage.tier0 import is_hard_reject
    except ImportError as exc:
        raise ImportError(
            "reward_triage is required when triage flags are on. "
            "Install with: pip install -e <path-to-reward-surrogate-research>"
        ) from exc
    return {
        "LabelCache": LabelCache,
        "make_cache_key": make_cache_key,
        "normalize_code": normalize_code,
        "PlateauStop": PlateauStop,
        "is_hard_reject": is_hard_reject,
        "run_tier0_on_code": run_tier0_on_code,
    }


def config_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def env_config_hash_from_args(args: Any) -> str:
    return config_hash(
        {
            "benchmark": getattr(args, "benchmark_name", None),
            "task": getattr(args, "task_name", None),
            "max_episode_steps": getattr(args, "max_episode_steps", None),
        }
    )


def train_config_hash_from_args(args: Any) -> str:
    return config_hash(
        {
            "train_max_steps": getattr(args, "train_max_steps", None),
            "eval_freq": getattr(args, "eval_freq", None),
            "train_num": getattr(args, "train_num", None),
            "n_eval_episodes": getattr(args, "n_eval_episodes", None),
            "seed": getattr(args, "seed", None),
            "train_seed_num": getattr(args, "train_seed_num", None),
        }
    )


def seeds_from_args(args: Any) -> List[int]:
    base = int(getattr(args, "seed", 1))
    n = int(getattr(args, "train_seed_num", 1))
    return list(range(base, base + max(1, n)))


@dataclass
class PreTrainDecision:
    skip_rl: bool = False
    cache_hit: bool = False
    validate_reject: bool = False
    cached_payload: Optional[Dict[str, Any]] = None
    reasons: List[str] = field(default_factory=list)
    steps_saved_est: int = 0
    cache_key: Optional[str] = None


@dataclass
class PostIterDecision:
    should_stop: bool = False
    shadow_score: Optional[float] = None
    steps_saved_est: int = 0
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def _cache_dir(flags: TriageFlags, run_dir: str) -> Path:
    if flags.cache_root:
        return Path(flags.cache_root)
    return Path(run_dir) / "triage_label_cache"


def _log_path(run_dir: str) -> Path:
    return Path(run_dir) / "triage_log.jsonl"


def append_triage_log(run_dir: str, record: Mapping[str, Any]) -> None:
    path = _log_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(record), default=str) + "\n")


def _plateau_state_path(run_dir: str) -> Path:
    return Path(run_dir) / "triage_plateau_state.json"


def _stop_flag_path(run_dir: str) -> Path:
    return Path(run_dir) / "triage_plateau_stop.json"


def before_train(
    flags: TriageFlags,
    *,
    reward_code: str,
    args: Any,
    run_dir: str,
    trajs: Optional[Sequence[Mapping[str, Any]]] = None,
    train_max_steps: Optional[int] = None,
) -> PreTrainDecision:
    """Seam: after generation / load, before TPE+train.

    Validate reject or cache hit can skip RL. TPE is still run by the caller
    unless the caller also skips the whole train block when ``skip_rl``.
    """
    if not flags.any_enabled:
        return PreTrainDecision()

    rt = _require_triage()
    decision = PreTrainDecision()
    steps = int(train_max_steps or getattr(args, "train_max_steps", 0) or 0)

    if flags.triage_validate:
        _exec_result, report = rt["run_tier0_on_code"](
            reward_code, trajs=trajs, allow_torch=False
        )
        decision.reasons.append(f"tier0:{report.status}")
        if rt["is_hard_reject"](report):
            decision.validate_reject = True
            decision.skip_rl = True
            decision.steps_saved_est = steps
            decision.reasons.append("validate_reject")
            append_triage_log(
                run_dir,
                {
                    "event": "validate_reject",
                    "reasons": decision.reasons,
                    "steps_saved_est": decision.steps_saved_est,
                },
            )
            return decision

    if flags.triage_cache:
        key = rt["make_cache_key"](
            reward_code,
            env_config_hash=env_config_hash_from_args(args),
            train_config_hash=train_config_hash_from_args(args),
            seeds=seeds_from_args(args),
        )
        decision.cache_key = key
        cache = rt["LabelCache"](_cache_dir(flags, run_dir))
        hit = cache.get_raw(key)
        if hit is not None and hit.get("learning_result") is not None:
            decision.cache_hit = True
            decision.skip_rl = True
            decision.cached_payload = hit
            decision.steps_saved_est = int(
                hit.get("train_steps_saved", steps) or steps
            )
            decision.reasons.append("cache_hit")
            append_triage_log(
                run_dir,
                {
                    "event": "cache_hit",
                    "cache_key": key,
                    "steps_saved_est": decision.steps_saved_est,
                },
            )
    return decision


def after_iteration(
    flags: TriageFlags,
    *,
    reward_code: str,
    args: Any,
    run_dir: str,
    iteration: int,
    rl_ran: bool,
    learning_result: Any,
    success_score: Optional[float],
    seed_successes: Optional[List[float]] = None,
    n_eval_episodes: Optional[int] = None,
    rolled_back: bool = False,
    archive_rollback_enabled: bool = False,
    train_max_steps: Optional[int] = None,
    cache_key: Optional[str] = None,
    env_metrics: Optional[Mapping[str, Any]] = None,
) -> PostIterDecision:
    """Seam: after each iteration (after CARD+ archive/rollback if any)."""
    if not flags.any_enabled:
        return PostIterDecision()

    rt = _require_triage()
    out = PostIterDecision()
    steps = int(train_max_steps or getattr(args, "train_max_steps", 0) or 0)
    metrics = dict(env_metrics or {})
    if success_score is not None:
        metrics.setdefault("success", float(success_score))
    out.metrics = metrics

    if flags.triage_cache and rl_ran and learning_result is not None:
        key = cache_key or rt["make_cache_key"](
            reward_code,
            env_config_hash=env_config_hash_from_args(args),
            train_config_hash=train_config_hash_from_args(args),
            seeds=seeds_from_args(args),
        )
        cache = rt["LabelCache"](_cache_dir(flags, run_dir))
        # learning_result may be numpy-heavy; store success curve only when possible
        curve = _extract_success_curve(learning_result)
        cache.put_raw(
            key,
            {
                "learning_result": {"successes": curve},
                "iteration": iteration,
                "metrics": metrics,
            },
            train_steps_saved=steps * max(1, int(getattr(args, "train_seed_num", 1))),
        )
        out.reasons.append("cache_put")

    if flags.triage_shadow:
        # Shadow only: log a trivial score from success if present; never filter.
        out.shadow_score = float(success_score) if success_score is not None else None
        out.reasons.append(f"shadow_score:{out.shadow_score}")

    if flags.triage_plateau_stop and success_score is not None:
        state_path = _plateau_state_path(run_dir)
        if state_path.exists():
            stopper = rt["PlateauStop"].from_dict(
                json.loads(state_path.read_text(encoding="utf-8"))
            )
        else:
            stopper = rt["PlateauStop"](
                margin=flags.plateau_margin,
                patience=flags.plateau_patience,
            )
        # Coexist with archive/rollback: if rollback is enabled, only stop after
        # a rollback attempt has been tried at least once in this run.
        rollback_ok = True
        if archive_rollback_enabled:
            rollback_ok = _rollback_attempted_once(run_dir) or rolled_back
            if rolled_back:
                _mark_rollback_attempted(run_dir)
        should_stop = stopper.update(
            float(success_score),
            seed_successes=seed_successes,
            n_eval_episodes=n_eval_episodes
            or int(getattr(args, "n_eval_episodes", 1) or 1),
            rollback_attempted=rollback_ok,
        )
        state_path.write_text(
            json.dumps(stopper.to_dict(), indent=2), encoding="utf-8"
        )
        out.should_stop = bool(should_stop)
        if should_stop:
            out.reasons.append("plateau_stop")
            # Remaining rounds not run → estimated save logged by runner
            _stop_flag_path(run_dir).write_text(
                json.dumps(
                    {
                        "stop": True,
                        "iteration": iteration,
                        "best": stopper.best,
                        "steps_per_round_est": steps,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    append_triage_log(
        run_dir,
        {
            "event": "after_iteration",
            "iteration": iteration,
            "rl_ran": rl_ran,
            "should_stop": out.should_stop,
            "shadow_score": out.shadow_score,
            "metrics": metrics,
            "reasons": out.reasons,
            "steps_saved_est": out.steps_saved_est,
        },
    )
    return out


def last_card_plus_rolled_back(run_dir: str) -> bool:
    """Read the latest CARD+ iteration log for ``rolled_back``."""
    path = Path(run_dir) / "card_plus_log.jsonl"
    if not path.exists():
        return False
    last = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if not last:
        return False
    try:
        return bool(json.loads(last).get("rolled_back"))
    except (TypeError, json.JSONDecodeError):
        return False


def plateau_stop_requested(run_dir: str) -> bool:
    path = _stop_flag_path(run_dir)
    if not path.exists():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("stop"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def _rollback_marker(run_dir: str) -> Path:
    return Path(run_dir) / "triage_rollback_attempted.json"


def _rollback_attempted_once(run_dir: str) -> bool:
    return _rollback_marker(run_dir).exists()


def _mark_rollback_attempted(run_dir: str) -> None:
    path = _rollback_marker(run_dir)
    path.write_text(json.dumps({"attempted": True}), encoding="utf-8")


def _extract_success_curve(learning_result: Any) -> List[float]:
    if learning_result is None:
        return []
    if isinstance(learning_result, dict):
        for key in ("successes", "results", "success_rate"):
            if key in learning_result:
                try:
                    return [float(x) for x in list(learning_result[key])]
                except (TypeError, ValueError):
                    pass
    return []


def score_from_learning_result(learning_result: Any, last_k: int = 3) -> Optional[float]:
    curve = _extract_success_curve(learning_result)
    if not curve:
        return None
    k = max(1, int(last_k))
    return float(sum(curve[-k:]) / len(curve[-k:]))
