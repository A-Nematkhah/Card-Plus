"""Offline dataset gate: keep only rewards that separate outcomes well.

Composite (calibrated)::
  composite = 0.10*offline_rank + 0.30*separation + 0.30*anti_hack
              + 0.30*intra_scenario

where
  offline_rank    = pair accuracies + progress_auc (often ~0.95–1; low weight)
  separation      = steep soft mean-gap between outcome returns
  anti_hack       = soft margins progress/success vs flee & orbit probes
  intra_scenario  = within-scenario Spearman(rule ranks, reward ranks)

Keep also requires pair floors, static anti-farm rules, and hard anti_hack probes.

Mode ``warn``: never skip RL; append report to feedback.
Mode ``gate``: skip RL when keep is False.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from card_plus.dataset.anti_hack_probes import run_anti_hack_probes
from card_plus.dataset.rescore import (
    PairAccuracy,
    RescoreReport,
    load_dataset_episodes,
    load_reward_fn_from_code,
    rescore_episodes_detailed,
)
from card_plus.dataset.reward_static import check_reward_code_static
from card_plus.dataset.scoring import (
    build_calibrated_metrics,
    offline_rank_score,
)

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]

# Module-level cache: one-step processes are short-lived; helps within a step.
_EPISODE_CACHE: Dict[str, list] = {}


@dataclass
class DatasetGateThresholds:
    min_success_collided: float = 0.85
    min_success_timeout: float = 0.80
    min_timeout_collided: float = 0.60
    min_progress_auc: float = 0.75
    # Calibrated for discriminative composite (low rank weight + intra/sep/hack).
    # Empirically: farm≈0.53, weak-but-passing LLM≈0.74, strong≈0.79–0.80.
    min_composite: float = 0.75
    min_offline_rank: float = 0.75  # legacy pair/AUC floor (evaluate_keep / diagnostics)
    min_anti_hack: float = 0.65
    min_separation: float = 0.70
    w_success_collided: float = 0.35
    w_success_timeout: float = 0.25
    w_timeout_collided: float = 0.25
    w_progress_auc: float = 0.15


@dataclass
class DatasetGateConfig:
    use_dataset_rescore: bool = False
    dataset_path: str = ""
    dataset_mode: str = "gate"  # warn | gate
    thresholds: DatasetGateThresholds = field(default_factory=DatasetGateThresholds)
    tpe_v1_threshold: float = 0.8  # reported only; not part of keep rule

    @property
    def enabled(self) -> bool:
        return bool(self.use_dataset_rescore and self.dataset_path)


def add_dataset_gate_arguments(parser: Any) -> None:
    g = parser.add_argument_group("dataset gate (CrowdNav offline)")
    g.add_argument(
        "--use_dataset_rescore",
        action="store_true",
        default=False,
        help="Rescore reward on a fixed CrowdNav dataset before/around RL",
    )
    g.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Dataset directory with index.json (e.g. results/datasets/crowdnav_ds_v1)",
    )
    g.add_argument(
        "--dataset_mode",
        type=str,
        default="gate",
        choices=("warn", "gate"),
        help="warn=feedback only; gate=skip RL when metrics fail",
    )
    g.add_argument("--dataset_min_success_collided", type=float, default=0.85)
    g.add_argument("--dataset_min_success_timeout", type=float, default=0.80)
    g.add_argument("--dataset_min_timeout_collided", type=float, default=0.60)
    g.add_argument("--dataset_min_progress_auc", type=float, default=0.75)
    g.add_argument(
        "--dataset_min_composite",
        type=float,
        default=0.75,
        help="Min calibrated composite (rank+sep+anti_hack+intra)",
    )
    g.add_argument("--dataset_min_anti_hack", type=float, default=0.65)
    g.add_argument("--dataset_min_separation", type=float, default=0.70)
    g.add_argument(
        "--llm_candidates",
        type=int,
        default=5,
        help="When dataset gate is on: generate this many passing rewards, then train the best",
    )
    g.add_argument(
        "--llm_candidate_retries",
        type=int,
        default=5,
        help="Max rewrite attempts per candidate slot after a gate reject",
    )


def dataset_gate_from_args(args: Any) -> DatasetGateConfig:
    # Fallbacks must match DatasetGateThresholds / argparse defaults (not softer).
    defaults = DatasetGateThresholds()
    th = DatasetGateThresholds(
        min_success_collided=float(
            getattr(args, "dataset_min_success_collided", defaults.min_success_collided)
        ),
        min_success_timeout=float(
            getattr(args, "dataset_min_success_timeout", defaults.min_success_timeout)
        ),
        min_timeout_collided=float(
            getattr(args, "dataset_min_timeout_collided", defaults.min_timeout_collided)
        ),
        min_progress_auc=float(
            getattr(args, "dataset_min_progress_auc", defaults.min_progress_auc)
        ),
        min_composite=float(getattr(args, "dataset_min_composite", defaults.min_composite)),
        min_anti_hack=float(getattr(args, "dataset_min_anti_hack", defaults.min_anti_hack)),
        min_separation=float(getattr(args, "dataset_min_separation", defaults.min_separation)),
    )
    return DatasetGateConfig(
        use_dataset_rescore=bool(getattr(args, "use_dataset_rescore", False)),
        dataset_path=str(getattr(args, "dataset_path", "") or ""),
        dataset_mode=str(getattr(args, "dataset_mode", "gate")),
        thresholds=th,
    )


def _pair_map(report: RescoreReport) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for pa in report.pair_accuracies:
        out[pa.pair] = float(pa.accuracy) if pa.n_pairs > 0 else float("nan")
    return out


def composite_score(report: RescoreReport, th: DatasetGateThresholds) -> float:
    """Backward-compatible name: returns offline_rank only (not full calibrated)."""
    return offline_rank_score(
        report,
        w_success_collided=th.w_success_collided,
        w_success_timeout=th.w_success_timeout,
        w_timeout_collided=th.w_timeout_collided,
        w_progress_auc=th.w_progress_auc,
    )


@dataclass
class DatasetGateResult:
    keep: bool
    skip_rl: bool
    mode: str
    composite: float
    reasons: List[str]
    report: RescoreReport
    metrics: Dict[str, float]

    def feedback_text(self) -> str:
        status = "PASS" if self.keep else "FAIL"
        lines = [
            f"Offline dataset gate ({self.mode}): {status}",
            f"  composite={self.composite:.4f} "
            f"(rank={self.metrics.get('offline_rank', float('nan')):.4f} "
            f"sep={self.metrics.get('separation', float('nan')):.4f} "
            f"anti_hack={self.metrics.get('anti_hack', float('nan')):.4f} "
            f"intra={self.metrics.get('intra_scenario', float('nan')):.4f})",
            f"  success/collided={self.metrics.get('success/collided', float('nan')):.4f}",
            f"  success/timeout={self.metrics.get('success/timeout', float('nan')):.4f}",
            f"  timeout/collided={self.metrics.get('timeout/collided', float('nan')):.4f}",
            f"  progress_auc={self.metrics.get('progress_auc', float('nan')):.4f}",
            f"  tpe_v1_rate={self.metrics.get('tpe_v1_rate', float('nan'))}",
        ]
        if "flee_mean" in self.metrics:
            lines.append(
                f"  probes: progress={self.metrics.get('progress_mean', float('nan')):.4f} "
                f"flee={self.metrics.get('flee_mean', float('nan')):.4f} "
                f"orbit={self.metrics.get('orbit_mean', float('nan')):.4f}"
            )
        if self.reasons:
            lines.append("  failures: " + "; ".join(self.reasons))
        if self.skip_rl:
            lines.append(
                "  RL skipped: reward did not meet calibrated keep criteria. "
                "Need strong outcome separation AND anti-hack margins "
                "(prefer goal progress over flee/orbit probes)."
            )
        return "\n".join(lines) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "keep": self.keep,
            "skip_rl": self.skip_rl,
            "mode": self.mode,
            "composite": self.composite,
            "reasons": list(self.reasons),
            "metrics": dict(self.metrics),
            "report": self.report.to_dict(),
        }


def evaluate_pair_floors(
    report: RescoreReport,
    th: Optional[DatasetGateThresholds] = None,
) -> tuple:
    """Check legacy pair/AUC floors. Return (ok, reasons, offline_rank, pair_metrics)."""
    th = th or DatasetGateThresholds()
    pairs = _pair_map(report)
    a = pairs.get("success/collided", float("nan"))
    b = pairs.get("success/timeout", float("nan"))
    c = pairs.get("timeout/collided", float("nan"))
    d = float(report.progress_auc)
    rank = composite_score(report, th)
    reasons: List[str] = []

    def _check(name: str, value: float, minimum: float) -> None:
        # Missing outcome pairs (NaN) must fail the floor — do not silently skip.
        if value != value:
            reasons.append(f"{name}=nan (missing outcome pair / undefined)")
            return
        if value < minimum:
            reasons.append(f"{name}={value:.4f} < {minimum:.2f}")

    _check("success/collided", a, th.min_success_collided)
    _check("success/timeout", b, th.min_success_timeout)
    _check("timeout/collided", c, th.min_timeout_collided)
    # progress_auc field = outcome return AUC (success vs fail); name kept for compat.
    _check("progress_auc", d, th.min_progress_auc)

    metrics = {
        "success/collided": a,
        "success/timeout": b,
        "timeout/collided": c,
        "progress_auc": d,
        "offline_rank": rank,
        "tpe_v1_rate": float("nan") if report.tpe_v1_rate is None else float(report.tpe_v1_rate),
    }
    return (len(reasons) == 0), reasons, rank, metrics


def _json_safe(obj: Any) -> Any:
    """Replace NaN/Inf so dataset_gate_*.json stays valid JSON."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


# Back-compat alias used by older tests/imports
def evaluate_keep(
    report: RescoreReport,
    th: Optional[DatasetGateThresholds] = None,
) -> tuple:
    """Pair/AUC keep check only (no probes). ``composite`` key = offline_rank."""
    ok, reasons, rank, metrics = evaluate_pair_floors(report, th)
    metrics = dict(metrics)
    metrics["composite"] = rank  # legacy callers expected this key as rank
    th = th or DatasetGateThresholds()
    floor = float(th.min_offline_rank)
    if rank < floor:
        # Legacy key name in metrics remains "composite" (== offline_rank here).
        reasons = list(reasons) + [f"offline_rank={rank:.4f} < {floor:.2f}"]
        ok = False
    return ok, reasons, rank, metrics


def _load_episodes_cached(dataset_path: PathLike) -> list:
    key = str(Path(dataset_path).resolve())
    if key not in _EPISODE_CACHE:
        logger.info("Loading dataset episodes from %s", key)
        _EPISODE_CACHE[key] = load_dataset_episodes(key)
    return _EPISODE_CACHE[key]


def run_dataset_gate(
    cfg: DatasetGateConfig,
    *,
    reward_code: Optional[str] = None,
    reward_path: Optional[PathLike] = None,
    run_dir: Optional[PathLike] = None,
    step: Optional[int] = None,
    report_suffix: Optional[str] = None,
) -> Optional[DatasetGateResult]:
    """Rescore reward on dataset and decide keep/skip. None if disabled."""
    if not cfg.enabled:
        return None
    path = Path(cfg.dataset_path)
    if not path.is_dir():
        raise FileNotFoundError(f"dataset_path not found: {path}")

    if reward_path is not None:
        code_text = Path(reward_path).read_text(encoding="utf-8")
    elif reward_code is not None:
        code_text = str(reward_code)
    else:
        raise ValueError("run_dataset_gate needs reward_code or reward_path")

    th = cfg.thresholds
    mode = str(cfg.dataset_mode).lower()
    static_ok, static_reasons = check_reward_code_static(code_text)
    if not static_ok:
        empty = RescoreReport(
            n_episodes=0,
            mean_return=0.0,
            outcome_counts={},
            progress_auc=0.0,
            tpe_v1_rate=None,
            pair_accuracies=[
                PairAccuracy("success/collided", 0, float("nan")),
                PairAccuracy("success/timeout", 0, float("nan")),
                PairAccuracy("timeout/collided", 0, float("nan")),
            ],
        )
        reasons = [f"static: {r}" for r in static_reasons]
        result = DatasetGateResult(
            keep=False,
            skip_rl=bool(mode == "gate"),
            mode=mode,
            composite=0.0,
            reasons=reasons,
            report=empty,
            metrics={"composite": 0.0, "offline_rank": 0.0, "separation": 0.0, "anti_hack": 0.0},
        )
        if run_dir is not None:
            out_dir = Path(run_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            tag = f"_{step}" if step is not None else ""
            if report_suffix:
                tag = f"{tag}_{report_suffix}"
            out = out_dir / f"dataset_gate{tag}.json"
            out.write_text(
                json.dumps(_json_safe(result.to_dict()), indent=2), encoding="utf-8"
            )
        print(
            f"[dataset_gate] keep=False skip_rl={result.skip_rl} composite=0.0000 "
            f"reasons={reasons}"
        )
        return result

    reward_fn = load_reward_fn_from_code(code_text)

    episodes = _load_episodes_cached(path)
    report, labeled, returns = rescore_episodes_detailed(
        episodes,
        reward_fn,
        threshold=cfg.tpe_v1_threshold,
        reward_code=code_text,
    )
    _pairs_ok, reasons, _rank, _pair_metrics = evaluate_pair_floors(report, th)

    probe = run_anti_hack_probes(reward_fn)
    if not probe.ok:
        reasons = list(reasons) + [f"anti_hack: {r}" for r in probe.reasons]

    composite, metrics = build_calibrated_metrics(
        report, labeled, returns, probe.metrics
    )

    if composite < th.min_composite:
        reasons.append(f"composite={composite:.4f} < {th.min_composite:.2f}")
    ah = metrics.get("anti_hack", float("nan"))
    if ah == ah and ah < th.min_anti_hack:
        reasons.append(f"anti_hack={ah:.4f} < {th.min_anti_hack:.2f}")
    sep = metrics.get("separation", float("nan"))
    if sep == sep and sep < th.min_separation:
        reasons.append(f"separation={sep:.4f} < {th.min_separation:.2f}")

    keep = len(reasons) == 0
    skip_rl = bool(mode == "gate" and not keep)
    result = DatasetGateResult(
        keep=keep,
        skip_rl=skip_rl,
        mode=mode,
        composite=composite,
        reasons=reasons,
        report=report,
        metrics=metrics,
    )

    if run_dir is not None:
        out_dir = Path(run_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"_{step}" if step is not None else ""
        if report_suffix:
            tag = f"{tag}_{report_suffix}"
        out = out_dir / f"dataset_gate{tag}.json"
        out.write_text(
            json.dumps(_json_safe(result.to_dict()), indent=2), encoding="utf-8"
        )
        logger.info("Wrote dataset gate report to %s", out)

    print(
        f"[dataset_gate] keep={keep} skip_rl={skip_rl} composite={composite:.4f} "
        f"(rank={metrics.get('offline_rank', float('nan')):.4f} "
        f"sep={metrics.get('separation', float('nan')):.4f} "
        f"anti_hack={metrics.get('anti_hack', float('nan')):.4f} "
        f"intra={metrics.get('intra_scenario', float('nan')):.4f}) "
        f"reasons={reasons or '-'}"
    )
    return result


def should_skip_rl_for_dataset_gate(result: Optional[DatasetGateResult]) -> bool:
    return bool(result is not None and result.skip_rl)


def load_dataset_gate_result_json(
    path: PathLike,
    *,
    mode: Optional[str] = None,
) -> Optional[DatasetGateResult]:
    """Reload a previously written ``dataset_gate_*.json`` (skip duplicate rescore)."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load dataset gate json %s: %s", p, exc)
        return None
    report_data = data.get("report") or {}
    pairs_raw = report_data.get("pair_accuracies") or []
    pair_accuracies = [
        PairAccuracy(
            pair=str(pa.get("pair", "")),
            n_pairs=int(pa.get("n_pairs", 0) or 0),
            accuracy=float(pa["accuracy"])
            if pa.get("accuracy") is not None
            else float("nan"),
        )
        for pa in pairs_raw
        if isinstance(pa, dict)
    ]
    report = RescoreReport(
        n_episodes=int(report_data.get("n_episodes", 0) or 0),
        mean_return=float(report_data.get("mean_return", 0.0) or 0.0),
        outcome_counts=dict(report_data.get("outcome_counts") or {}),
        progress_auc=float(report_data["progress_auc"])
        if report_data.get("progress_auc") is not None
        else float("nan"),
        tpe_v1_rate=(
            None
            if report_data.get("tpe_v1_rate") is None
            else float(report_data["tpe_v1_rate"])
        ),
        pair_accuracies=pair_accuracies,
        extras=dict(report_data.get("extras") or {}),
    )
    mode_s = str(mode or data.get("mode") or "gate")
    keep = bool(data.get("keep", False))
    skip_rl = bool(data.get("skip_rl", mode_s == "gate" and not keep))
    metrics = {str(k): float(v) if v is not None else float("nan") for k, v in (data.get("metrics") or {}).items()}
    return DatasetGateResult(
        keep=keep,
        skip_rl=skip_rl,
        mode=mode_s,
        composite=float(data.get("composite", 0.0) or 0.0),
        reasons=[str(r) for r in (data.get("reasons") or [])],
        report=report,
        metrics=metrics,
    )
