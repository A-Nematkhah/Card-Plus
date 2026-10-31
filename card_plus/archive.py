"""Reward version archive + rollback notices."""

from __future__ import annotations

import difflib
import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from card_plus.config import CardPlusConfig

logger = logging.getLogger(__name__)

def unified_diff_line_count(a: str, b: str) -> int:
    """Number of lines in a unified diff from ``a`` to ``b`` (0 if identical)."""
    if a == b:
        return 0
    lines = list(
        difflib.unified_diff(
            a.splitlines(),
            b.splitlines(),
            lineterm="",
        )
    )
    return int(len(lines))


ROLLBACK_NOTICE_TEMPLATE = """# Rollback Notice
Your last modification reduced the success rate from {best_score:.0%} to {score:.0%}, so the reward function was reverted to version {best_idx}. Below is the diff between the best version and the failed one (best -> failed):
{unified_diff}
Base your next version on the best version below and do NOT repeat the changes that were harmful.
{best_code}
"""


@dataclass
class VersionRecord:
    iteration: int
    reward_code: str
    sub_reward_names: List[str] = field(default_factory=list)
    rl_ran: bool = False
    success_curve: List[float] = field(default_factory=list)
    version_score: Optional[float] = None
    curve_auc: Optional[float] = None
    mean_return: Optional[float] = None
    tpe_acc_v1: Optional[float] = None
    tpe_pass_v1: Optional[bool] = None
    rho_full: Optional[float] = None
    rho_dense: Optional[float] = None
    hacked: Optional[bool] = None
    verdict: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VersionRecord":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def success_curve_auc(curve: Sequence[float]) -> float:
    """Normalized trapezoid AUC of a success-rate curve in [0,1]."""
    y = np.asarray(list(curve), dtype=np.float64).ravel()
    if y.size == 0:
        return 0.0
    if y.size == 1:
        return float(y[0])
    x = np.linspace(0.0, 1.0, num=y.size)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(y, x))


def score_from_curve(curve: Sequence[float], last_k: int) -> Optional[float]:
    y = list(curve)
    if not y:
        return None
    k = max(1, int(last_k))
    return float(np.mean(y[-k:]))


class RewardArchive:
    def __init__(self, run_dir: Optional[str] = None, cfg: Optional[CardPlusConfig] = None) -> None:
        self.run_dir = Path(run_dir) if run_dir else None
        self.cfg = cfg or CardPlusConfig()
        self.versions: List[VersionRecord] = []
        self._path = (self.run_dir / "card_plus_archive.json") if self.run_dir else None
        if self._path and self._path.exists():
            self.load(self._path)

    def add(self, record: VersionRecord) -> None:
        # Replace same iteration if re-run
        self.versions = [v for v in self.versions if v.iteration != record.iteration]
        self.versions.append(record)
        self.versions.sort(key=lambda v: v.iteration)
        self.save()

    def best_index(self) -> Optional[int]:
        scored = [v for v in self.versions if v.rl_ran and v.version_score is not None]
        if not scored:
            return None

        def key(v: VersionRecord) -> tuple:
            return (
                float(v.version_score or 0.0),
                float(v.curve_auc or 0.0),
                float(v.rho_full or 0.0),
            )

        best = max(scored, key=key)
        return best.iteration

    def best_record(self) -> Optional[VersionRecord]:
        idx = self.best_index()
        if idx is None:
            return None
        for v in self.versions:
            if v.iteration == idx:
                return v
        return None

    @property
    def best_code(self) -> Optional[str]:
        rec = self.best_record()
        return rec.reward_code if rec else None

    def best_score(self) -> Optional[float]:
        rec = self.best_record()
        return rec.version_score if rec else None

    def should_rollback(self, score: Optional[float]) -> bool:
        if not self.cfg.use_archive_rollback:
            return False
        if score is None:
            return False
        best = self.best_score()
        if best is None:
            return False
        # Rollback if current is worse than best by more than margin
        # (compare against previous best excluding current — caller should add current after check,
        #  or we compare against best among *other* versions)
        return score < best - self.cfg.rollback_margin

    def should_rollback_against(
        self, score: Optional[float], exclude_iteration: Optional[int] = None
    ) -> bool:
        if not self.cfg.use_archive_rollback or score is None:
            return False
        others = [
            v
            for v in self.versions
            if v.rl_ran
            and v.version_score is not None
            and (exclude_iteration is None or v.iteration != exclude_iteration)
        ]
        if not others:
            return False

        def key(v: VersionRecord) -> tuple:
            return (
                float(v.version_score or 0.0),
                float(v.curve_auc or 0.0),
                float(v.rho_full or 0.0),
            )

        best = max(others, key=key)
        return score < float(best.version_score) - self.cfg.rollback_margin

    def format_rollback_notice(
        self,
        failed: VersionRecord,
        best: VersionRecord,
    ) -> str:
        diff_lines = list(
            difflib.unified_diff(
                best.reward_code.splitlines(),
                failed.reward_code.splitlines(),
                fromfile=f"version_{best.iteration}",
                tofile=f"version_{failed.iteration}",
                lineterm="",
            )
        )
        max_lines = self.cfg.max_diff_lines
        if len(diff_lines) > max_lines:
            diff_lines = diff_lines[:max_lines] + [f"... ({len(diff_lines) - max_lines} more lines truncated)"]
        unified = "\n".join(diff_lines) if diff_lines else "(no textual diff)"
        return ROLLBACK_NOTICE_TEMPLATE.format(
            best_score=float(best.version_score or 0.0),
            score=float(failed.version_score or 0.0),
            best_idx=best.iteration,
            unified_diff=unified,
            best_code=best.reward_code,
        )

    def save(self, path: Optional[Path] = None) -> None:
        path = path or self._path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "versions": [v.to_dict() for v in self.versions],
            "best_iteration": self.best_index(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.versions = [VersionRecord.from_dict(v) for v in data.get("versions", [])]


def extract_success_curve(learning_result: Any) -> List[float]:
    """Best-effort extraction of eval success rates from train_and_eval payload.

    SB3 ``evaluations.npz`` stores ``successes`` as (n_evals, n_episodes); we
    average over episodes per eval to get a 1-D success-rate curve.
    """
    if learning_result is None:
        return []
    if isinstance(learning_result, dict):
        for key in ("successes", "success_rate", "eval_successes", "success_curve"):
            if key in learning_result and learning_result[key] is not None:
                arr = np.asarray(learning_result[key], dtype=np.float64)
                if arr.ndim == 2:
                    return [float(np.mean(row)) for row in arr]
                return [float(x) for x in arr.ravel()]
        ev = learning_result.get("evaluations")
        if isinstance(ev, dict) and "successes" in ev:
            arr = np.asarray(ev["successes"], dtype=np.float64)
            if arr.ndim == 2:
                return [float(np.mean(row)) for row in arr]
            return [float(x) for x in arr.ravel()]
    return []


def extract_mean_return(learning_result: Any) -> Optional[float]:
    if not isinstance(learning_result, dict):
        return None
    for key in ("mean_return", "mean_reward", "episode_reward"):
        if key in learning_result and learning_result[key] is not None:
            try:
                return float(np.mean(learning_result[key]))
            except Exception:
                try:
                    return float(learning_result[key])
                except Exception:
                    return None
    return None
