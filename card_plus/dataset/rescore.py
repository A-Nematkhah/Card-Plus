"""Offline scoring of reward callables / LLM ``specific_*.py`` on stored episodes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from card_plus.dataset.labels import label_episode
from card_plus.dataset.schema import Episode, load_episode

PathLike = Union[str, Path]
RewardFn = Callable[[np.ndarray, np.ndarray], Any]  # (action, obs) -> float | (float, dict)

# Parallel rescore is opt-in (CARD_GATE_WORKERS>1). Default serial: packing +
# vectorized pairwise already make one gate pass ~sub-second; LLM dominates wall time.
_DEFAULT_GATE_WORKERS = 1


@dataclass
class PairAccuracy:
    pair: str
    n_pairs: int
    accuracy: float


@dataclass
class RescoreReport:
    n_episodes: int
    mean_return: float
    outcome_counts: Dict[str, int]
    progress_auc: float
    tpe_v1_rate: Optional[float] = None
    pair_accuracies: List[PairAccuracy] = field(default_factory=list)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_episodes": self.n_episodes,
            "mean_return": self.mean_return,
            "outcome_counts": dict(self.outcome_counts),
            "progress_auc": self.progress_auc,
            "tpe_v1_rate": self.tpe_v1_rate,
            "pair_accuracies": [pa.__dict__ for pa in self.pair_accuracies],
            "extras": dict(self.extras),
        }


def load_reward_fn_from_code(code: str) -> RewardFn:
    """Exec a CARD-style ``compute_dense_reward(self, action, obs)`` snippet.

    The bound ``self`` exposes ``pre_obs`` (previous flat obs) so rewards can use
    delta goal-distance. Callers may set it via ``fn.set_pre_obs(obs)``.
    """
    namespace: Dict[str, Any] = {"np": np, "Dict": Dict, "Tuple": Tuple}
    exec(code, namespace)
    if "compute_dense_reward" not in namespace:
        raise ValueError("Code must define compute_dense_reward(self, action, obs).")
    fn = namespace["compute_dense_reward"]

    class _Holder:
        pre_obs = None

    holder = _Holder()

    def set_pre_obs(obs: Optional[np.ndarray]) -> None:
        holder.pre_obs = None if obs is None else np.asarray(obs, dtype=np.float64)

    def reset_pre_obs() -> None:
        holder.pre_obs = None

    def _call(action: np.ndarray, obs: np.ndarray) -> Any:
        obs_arr = np.asarray(obs, dtype=np.float64)
        if holder.pre_obs is None:
            holder.pre_obs = obs_arr
        return fn(holder, action, obs_arr)

    _call.set_pre_obs = set_pre_obs  # type: ignore[attr-defined]
    _call.reset_pre_obs = reset_pre_obs  # type: ignore[attr-defined]
    return _call


def load_reward_fn_from_path(path: PathLike) -> RewardFn:
    return load_reward_fn_from_code(Path(path).read_text(encoding="utf-8"))


def _scalar_reward(result: Any) -> float:
    if isinstance(result, tuple):
        return float(result[0])
    return float(result)


def ensure_episode_gate_arrays(episode: Episode) -> None:
    """Materialize stacked action/obs arrays once (mutates episode attributes)."""
    if getattr(episode, "_gate_actions", None) is not None:
        if episode.labels is None:
            episode.labels = label_episode(episode)
        return
    episode._gate_actions = episode.actions_array()  # type: ignore[attr-defined]
    episode._gate_obs = episode.obs_array()  # type: ignore[attr-defined]
    if episode.initial_obs is None:
        episode._gate_initial = None  # type: ignore[attr-defined]
    else:
        episode._gate_initial = np.asarray(episode.initial_obs, dtype=np.float64)  # type: ignore[attr-defined]
    if episode.labels is None:
        episode.labels = label_episode(episode)


def score_episode_arrays(
    reward_fn: RewardFn,
    actions: np.ndarray,
    obs: np.ndarray,
    initial_obs: Optional[np.ndarray],
) -> np.ndarray:
    """Tight step loop over prestacked arrays (avoids StepRecord attribute churn)."""
    reset = getattr(reward_fn, "reset_pre_obs", None)
    set_pre = getattr(reward_fn, "set_pre_obs", None)
    if reset is not None:
        reset()
    n = int(actions.shape[0])
    out = np.empty(n, dtype=np.float64)
    prev = None if initial_obs is None else initial_obs
    for t in range(n):
        obs_t = obs[t]
        if set_pre is not None:
            set_pre(prev if prev is not None else obs_t)
        out[t] = _scalar_reward(reward_fn(actions[t], obs_t))
        prev = obs_t
    return out


def score_episode(episode: Episode, reward_fn: RewardFn) -> np.ndarray:
    """Re-score each stored step with ``reward_fn(action, post_step_obs)``.

    Sets ``pre_obs`` from ``episode.initial_obs`` then prior step obs when the
    reward callable supports ``set_pre_obs`` (delta-progress rewards).
    """
    ensure_episode_gate_arrays(episode)
    return score_episode_arrays(
        reward_fn,
        episode._gate_actions,  # type: ignore[attr-defined]
        episode._gate_obs,  # type: ignore[attr-defined]
        episode._gate_initial,  # type: ignore[attr-defined]
    )


def _gate_workers() -> int:
    raw = os.environ.get("CARD_GATE_WORKERS", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            return _DEFAULT_GATE_WORKERS
    return _DEFAULT_GATE_WORKERS


def _score_pack_chunk(
    payload: Tuple[str, List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]],
) -> List[float]:
    """ProcessPool worker: compile reward once, score a chunk of packed episodes."""
    code, packs = payload
    reward_fn = load_reward_fn_from_code(code)
    out: List[float] = []
    for actions, obs, initial in packs:
        out.append(float(np.sum(score_episode_arrays(reward_fn, actions, obs, initial))))
    return out


def episode_returns_for_reward(
    episodes: Sequence[Episode],
    reward_fn: RewardFn,
    *,
    reward_code: Optional[str] = None,
    workers: Optional[int] = None,
) -> List[float]:
    """Score all episodes; optionally parallelize across processes when code is given."""
    for ep in episodes:
        ensure_episode_gate_arrays(ep)
    n_workers = _DEFAULT_GATE_WORKERS if workers is None else int(workers)
    use_pool = (
        reward_code is not None
        and n_workers > 1
        and len(episodes) >= 32
    )
    if not use_pool:
        return [
            float(np.sum(score_episode(ep, reward_fn)))
            for ep in episodes
        ]

    packs = [
        (
            ep._gate_actions,  # type: ignore[attr-defined]
            ep._gate_obs,  # type: ignore[attr-defined]
            ep._gate_initial,  # type: ignore[attr-defined]
        )
        for ep in episodes
    ]
    # Chunk to cut IPC overhead vs one task per episode.
    n_chunks = min(n_workers * 2, len(packs))
    chunk_size = max(1, (len(packs) + n_chunks - 1) // n_chunks)
    payloads = []
    for i in range(0, len(packs), chunk_size):
        payloads.append((str(reward_code), packs[i : i + chunk_size]))
    try:
        returns: List[float] = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for part in pool.map(_score_pack_chunk, payloads):
                returns.extend(part)
        return returns
    except Exception:
        # Windows spawn / nested __main__ issues → safe serial fallback.
        return [
            float(np.sum(score_episode(ep, reward_fn)))
            for ep in episodes
        ]


def _episode_return(rewards: np.ndarray) -> float:
    """Episode score aligned with RL: cumulative (sum) return, not mean step reward."""
    if rewards.size == 0:
        return 0.0
    return float(np.sum(rewards))


# Back-compat name; implementation is sum (matches online SAC objective).
_episode_mean_return = _episode_return


def _outcomes_and_returns(
    episodes: Sequence[Episode],
    returns: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    outs = np.asarray(
        [(ep.labels or label_episode(ep)).outcome for ep in episodes],
        dtype=object,
    )
    rets = np.asarray(list(returns), dtype=np.float64)
    return outs, rets


def _mann_whitney_auc(better: np.ndarray, worse: np.ndarray) -> float:
    """P(better > worse) + 0.5 P(tie), vectorized."""
    if better.size == 0 or worse.size == 0:
        return float("nan")
    # Broadcasting: [n_better, n_worse]
    gt = better[:, None] > worse[None, :]
    eq = better[:, None] == worse[None, :]
    return float((gt.astype(np.float64) + 0.5 * eq.astype(np.float64)).mean())


def outcome_return_auc(episodes: Sequence[Episode], returns: Sequence[float]) -> float:
    """Mann–Whitney AUC on episode returns: P(return_success > return_fail) + 0.5 ties.

    Despite the historical field name ``progress_auc`` on ``RescoreReport``, this is
    **not** an AUC of the task-progress measure — it ranks episode returns by outcome.
    Returns are cumulative (sum) rewards, matching the online RL objective.
    """
    outs, rets = _outcomes_and_returns(episodes, returns)
    succ = rets[outs == "success"]
    fail = rets[outs != "success"]
    return _mann_whitney_auc(succ, fail)


# Back-compat alias (name was misleading; prefer outcome_return_auc).
_progress_auc = outcome_return_auc


def _pair_accuracy(
    episodes: Sequence[Episode],
    returns: Sequence[float],
    a: str,
    b: str,
) -> PairAccuracy:
    """Prefer higher return on outcome ``a`` vs ``b`` (CARD preference style).

    Ties count as 0.5 (same policy as ``outcome_return_auc``).
    """
    outs, rets = _outcomes_and_returns(episodes, returns)
    group_a = rets[outs == a]
    group_b = rets[outs == b]
    n = int(group_a.size * group_b.size)
    if n == 0:
        return PairAccuracy(pair=f"{a}/{b}", n_pairs=0, accuracy=float("nan"))
    acc = _mann_whitney_auc(group_a, group_b)
    return PairAccuracy(pair=f"{a}/{b}", n_pairs=n, accuracy=acc)


def tpe_v1_rate(
    episodes: Sequence[Episode],
    returns: Sequence[float],
    *,
    threshold: float = 0.8,
) -> Optional[float]:
    """Fraction of successful episodes whose return beats max failed return.

    Mirrors this repo's ``trajectories_return_check`` style (not pairwise γ-TPE).
    """
    outs, rets = _outcomes_and_returns(episodes, returns)
    succ = rets[outs == "success"]
    fail = rets[outs != "success"]
    if succ.size == 0:
        return None
    if fail.size == 0:
        return 1.0
    worst_fail = float(fail.max())
    rate = float(np.mean(succ > worst_fail))
    # threshold is reported by caller; here we return the raw rate
    del threshold
    return rate


def _finalize_report(
    labeled: Sequence[Episode],
    returns: Sequence[float],
    *,
    threshold: float,
    extras: Optional[Dict[str, Any]] = None,
) -> RescoreReport:
    counts: Dict[str, int] = {}
    for ep in labeled:
        assert ep.labels is not None
        counts[ep.labels.outcome] = counts.get(ep.labels.outcome, 0) + 1

    outs, rets = _outcomes_and_returns(labeled, returns)
    pairs = []
    for a, b in (("success", "collided"), ("success", "timeout"), ("timeout", "collided")):
        group_a = rets[outs == a]
        group_b = rets[outs == b]
        n = int(group_a.size * group_b.size)
        pairs.append(
            PairAccuracy(
                pair=f"{a}/{b}",
                n_pairs=n,
                accuracy=_mann_whitney_auc(group_a, group_b) if n else float("nan"),
            )
        )
    succ = rets[outs == "success"]
    fail = rets[outs != "success"]
    rate = None
    if succ.size == 0:
        rate = None
    elif fail.size == 0:
        rate = 1.0
    else:
        rate = float(np.mean(succ > float(fail.max())))
    ret_auc = _mann_whitney_auc(succ, fail)
    # Prefer label progress (already computed on load) over restacking obs arrays.
    progress_vals = [
        float(getattr(ep.labels, "progress", 0.0) or 0.0)
        for ep in labeled
        if ep.labels is not None
    ]
    extra = {
        "threshold": threshold,
        "tpe_v1_pass": None if rate is None else bool(rate >= threshold),
        "mean_progress": float(np.mean(progress_vals)) if progress_vals else 0.0,
        "outcome_return_auc": ret_auc,
        "progress_auc_meaning": "outcome_return_auc(success vs fail cumulative returns)",
        "episode_return": "sum",
    }
    if extras:
        extra.update(extras)
    return RescoreReport(
        n_episodes=len(labeled),
        mean_return=float(np.mean(returns)) if returns else 0.0,
        outcome_counts=counts,
        progress_auc=ret_auc,
        tpe_v1_rate=rate,
        pair_accuracies=pairs,
        extras=extra,
    )


def rescore_episodes(
    episodes: Sequence[Episode],
    reward_fn: RewardFn,
    *,
    threshold: float = 0.8,
    reward_code: Optional[str] = None,
    workers: Optional[int] = None,
) -> RescoreReport:
    report, _labeled, _returns = rescore_episodes_detailed(
        episodes,
        reward_fn,
        threshold=threshold,
        reward_code=reward_code,
        workers=workers,
    )
    return report


def rescore_episodes_detailed(
    episodes: Sequence[Episode],
    reward_fn: RewardFn,
    *,
    threshold: float = 0.8,
    reward_code: Optional[str] = None,
    workers: Optional[int] = None,
) -> Tuple[RescoreReport, List[Episode], List[float]]:
    """Like ``rescore_episodes`` but also returns labeled episodes and cumulative returns."""
    labeled: List[Episode] = list(episodes)
    for ep in labeled:
        ensure_episode_gate_arrays(ep)
    returns = episode_returns_for_reward(
        labeled, reward_fn, reward_code=reward_code, workers=workers
    )
    return _finalize_report(labeled, returns, threshold=threshold), labeled, returns


def rescore_episodes_stored_gt(
    episodes: Sequence[Episode],
    *,
    threshold: float = 0.8,
) -> RescoreReport:
    """Use stored ``reward_raw`` cumulative returns (simulator GT) — no (action, obs) re-score."""
    labeled = []
    returns = []
    for ep in episodes:
        if ep.labels is None:
            ep.labels = label_episode(ep)
        labeled.append(ep)
        returns.append(_episode_return(ep.rewards_raw_array()))
    return _finalize_report(
        labeled, returns, threshold=threshold, extras={"reward": "gt_stored"}
    )


def rescore_paths(
    episode_paths: Sequence[PathLike],
    reward_fn: RewardFn,
    **kwargs: Any,
) -> RescoreReport:
    episodes = [load_episode(p) for p in episode_paths]
    return rescore_episodes(episodes, reward_fn, **kwargs)


def load_dataset_episodes(dataset_dir: PathLike) -> List[Episode]:
    """Load all episodes listed in ``index.json`` (or every ``episodes/*`` file)."""
    root = Path(dataset_dir)
    index_path = root / "index.json"
    paths: List[Path] = []
    if index_path.is_file():
        rows = json.loads(index_path.read_text(encoding="utf-8"))
        for row in rows:
            paths.append(root / row["path"])
    else:
        ep_dir = root / "episodes"
        paths = sorted(ep_dir.glob("*.pkl")) + sorted(ep_dir.glob("*.npz"))
    episodes = [load_episode(p) for p in paths]
    # Warm gate arrays/labels once so candidate retries avoid repeated stacking.
    for ep in episodes:
        ensure_episode_gate_arrays(ep)
    return episodes


def builtin_reward_fn(name: str) -> RewardFn:
    key = str(name).strip().lower()
    if key in ("progress", "neg_goal_dist"):
        def _progress(action: np.ndarray, obs: np.ndarray) -> float:
            del action
            return -float(np.linalg.norm(obs[0:2] - obs[4:6]))

        return _progress
    if key in ("sparse_success", "hack_sparse"):
        # Sparse terminal-only shaping proxy from obs (goal proximity).
        def _sparse(action: np.ndarray, obs: np.ndarray) -> float:
            del action
            dist = float(np.linalg.norm(obs[0:2] - obs[4:6]))
            return 10.0 if dist < float(obs[6]) else 0.0

        return _sparse
    if key in ("constant", "zero"):
        def _zero(action: np.ndarray, obs: np.ndarray) -> float:
            del action, obs
            return 0.0

        return _zero
    raise KeyError(
        f"Unknown builtin reward {name!r}. Use: progress, sparse_success, constant — "
        "or --reward gt / --reward-file"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    p = argparse.ArgumentParser(description="Rescore a collected CrowdNav dataset")
    p.add_argument("--dataset", type=str, required=True, help="Dataset dir (with index.json)")
    p.add_argument(
        "--reward",
        type=str,
        default="gt",
        help="gt | progress | sparse_success | constant  (or use --reward-file)",
    )
    p.add_argument(
        "--reward-file",
        type=str,
        default=None,
        help="Path to CARD specific_*.py defining compute_dense_reward",
    )
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write JSON report (default: <dataset>/rescore_<reward>.json)",
    )
    args = p.parse_args(argv)

    dataset = Path(args.dataset)
    print(f"[rescore] loading {dataset}")
    episodes = load_dataset_episodes(dataset)
    print(f"[rescore] {len(episodes)} episodes")

    if args.reward_file:
        reward_fn = load_reward_fn_from_path(args.reward_file)
        report = rescore_episodes(episodes, reward_fn, threshold=float(args.threshold))
        tag = Path(args.reward_file).stem
        report.extras["reward"] = f"file:{args.reward_file}"
    elif str(args.reward).lower() in ("gt", "ground_truth", "reward_raw"):
        report = rescore_episodes_stored_gt(episodes, threshold=float(args.threshold))
        tag = "gt"
    else:
        reward_fn = builtin_reward_fn(args.reward)
        report = rescore_episodes(episodes, reward_fn, threshold=float(args.threshold))
        tag = str(args.reward).lower()
        report.extras["reward"] = tag

    out = Path(args.out) if args.out else dataset / f"rescore_{tag}.json"
    out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    print(f"[rescore] mean_return={report.mean_return:.4f}")
    print(f"[rescore] outcomes={report.outcome_counts}")
    print(f"[rescore] progress_auc={report.progress_auc}")
    print(f"[rescore] tpe_v1_rate={report.tpe_v1_rate} pass={report.extras.get('tpe_v1_pass')}")
    for pa in report.pair_accuracies:
        print(f"[rescore] pair {pa.pair}: acc={pa.accuracy} n={pa.n_pairs}")
    print(f"[rescore] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
