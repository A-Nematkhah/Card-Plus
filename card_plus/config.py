"""CARD+ configuration (all flags default False for baseline reproducibility)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Mapping, Optional


@dataclass
class CardPlusConfig:
    use_archive_rollback: bool = False
    use_subreward_stats: bool = False
    use_tpe_v2: bool = False
    # archive / rollback
    score_last_k_evals: int = 3
    rollback_margin: float = 0.05
    score_final_version: bool = True
    max_diff_lines: int = 60
    # Last train step of this run (no further Coder query follows). None = unknown.
    card_plus_max_step: Optional[int] = None
    # sub-reward stats
    dominant_share: float = 0.6
    sparse_nonzero_frac: float = 0.05
    inert_std_eps: float = 1e-6
    neg_corr_threshold: float = -0.2
    max_feedback_lines: int = 25
    # tpe v2
    tpe_v2_mode: str = "warn"  # "warn" | "gate"
    tpe_v2_min_trajs: int = 10
    tpe_v2_tau_dense: float = 0.3
    progress_validation_auc_min: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def any_enabled(self) -> bool:
        return bool(
            self.use_archive_rollback
            or self.use_subreward_stats
            or self.use_tpe_v2
        )


def card_plus_from_args(args: Any) -> CardPlusConfig:
    """Build CardPlusConfig from an argparse Namespace (missing attrs → defaults)."""
    kwargs: Dict[str, Any] = {}
    for f in fields(CardPlusConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    return CardPlusConfig(**kwargs)


def add_card_plus_arguments(parser: Any) -> None:
    """Register CARD+ CLI flags on an argparse parser (defaults keep baseline)."""
    g = parser.add_argument_group("CARD+")
    g.add_argument("--use_archive_rollback", action="store_true", default=False)
    g.add_argument("--use_subreward_stats", action="store_true", default=False)
    g.add_argument("--use_tpe_v2", action="store_true", default=False)
    g.add_argument("--score_last_k_evals", type=int, default=3)
    g.add_argument("--rollback_margin", type=float, default=0.05)
    g.add_argument(
        "--score_final_version",
        action="store_true",
        default=True,
        help="Train last untrained version once so it can be scored (default True).",
    )
    g.add_argument(
        "--no_score_final_version",
        action="store_false",
        dest="score_final_version",
        help="Disable final short training for unscored last version.",
    )
    g.add_argument("--max_diff_lines", type=int, default=60)
    g.add_argument(
        "--card_plus_max_step",
        type=int,
        default=None,
        help="Last train step of the CARD loop (no further Coder query follows). "
        "Used for score_final_version and to suppress rollback on the last iteration.",
    )
    g.add_argument("--dominant_share", type=float, default=0.6)
    g.add_argument("--sparse_nonzero_frac", type=float, default=0.05)
    g.add_argument("--inert_std_eps", type=float, default=1e-6)
    g.add_argument("--neg_corr_threshold", type=float, default=-0.2)
    g.add_argument("--max_feedback_lines", type=int, default=25)
    g.add_argument("--tpe_v2_mode", type=str, default="warn", choices=["warn", "gate"])
    g.add_argument("--tpe_v2_min_trajs", type=int, default=10)
    g.add_argument("--tpe_v2_tau_dense", type=float, default=0.3)
    g.add_argument("--progress_validation_auc_min", type=float, default=0.9)


def merge_card_plus_dict(cfg: CardPlusConfig, overrides: Optional[Mapping[str, Any]]) -> CardPlusConfig:
    if not overrides:
        return cfg
    data = cfg.to_dict()
    data.update({k: v for k, v in overrides.items() if k in data})
    return CardPlusConfig(**data)
