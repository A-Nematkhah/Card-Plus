"""Train/val scenario splits for offline dataset gate (anti-overfit)."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from card_plus.dataset.schema import Episode

PathLike = Union[str, Path]

SPLIT_FILENAME = "scenario_split.json"


def _scenario_id(ep: Episode) -> str:
    return str(ep.scenario.scenario_id)


def _difficulty_key(ep: Episode) -> float:
    d = ep.scenario.difficulty or {}
    if "humans_near_path" in d:
        return float(d["humans_near_path"])
    return 0.0


def unique_scenario_ids(episodes: Sequence[Episode]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for ep in episodes:
        sid = _scenario_id(ep)
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def stratified_train_val_ids(
    episodes: Sequence[Episode],
    *,
    val_fraction: float = 0.25,
    seed: int = 0,
    n_bins: int = 4,
) -> Dict[str, List[str]]:
    """Split unique scenario_ids into train/val, stratified by difficulty bins."""
    by_id: Dict[str, Episode] = {}
    for ep in episodes:
        sid = _scenario_id(ep)
        if sid not in by_id:
            by_id[sid] = ep
    ids = sorted(by_id.keys())
    if not ids:
        return {"train": [], "val": [], "seed": seed, "val_fraction": val_fraction}

    diffs = np.asarray([_difficulty_key(by_id[i]) for i in ids], dtype=np.float64)
    # Quantile bins (stable even with ties).
    try:
        edges = np.unique(np.quantile(diffs, np.linspace(0.0, 1.0, int(n_bins) + 1)))
    except Exception:
        edges = np.unique(diffs)
    if edges.size < 2:
        bin_idx = np.zeros(len(ids), dtype=np.int64)
    else:
        bin_idx = np.digitize(diffs, edges[1:-1], right=False)

    rng = np.random.RandomState(int(seed))
    train: List[str] = []
    val: List[str] = []
    for b in sorted(set(int(x) for x in bin_idx)):
        members = [ids[i] for i in range(len(ids)) if int(bin_idx[i]) == b]
        rng.shuffle(members)
        n_val = int(round(len(members) * float(val_fraction)))
        if len(members) >= 2:
            n_val = max(1, min(len(members) - 1, n_val))
        else:
            n_val = 0
        val.extend(members[:n_val])
        train.extend(members[n_val:])
    return {
        "train": sorted(train),
        "val": sorted(val),
        "seed": int(seed),
        "val_fraction": float(val_fraction),
        "n_bins": int(n_bins),
    }


def write_split(split: Dict[str, Any], path: PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return p


def load_split(path: PathLike) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_dataset_split(
    dataset_dir: PathLike,
    episodes: Sequence[Episode],
    *,
    val_fraction: float = 0.25,
    seed: int = 0,
    force: bool = False,
) -> Dict[str, Any]:
    """Load or create ``scenario_split.json`` next to the dataset."""
    root = Path(dataset_dir)
    path = root / SPLIT_FILENAME
    if path.is_file() and not force:
        return load_split(path)
    split = stratified_train_val_ids(
        episodes, val_fraction=val_fraction, seed=seed
    )
    write_split(split, path)
    return split


def filter_episodes_by_split(
    episodes: Sequence[Episode],
    split: Dict[str, Any],
    which: str,
) -> List[Episode]:
    """``which`` in {train, val, all}."""
    key = str(which).lower()
    if key in ("all", "*", "full"):
        return list(episodes)
    allowed = set(split.get(key) or [])
    return [ep for ep in episodes if _scenario_id(ep) in allowed]


def difficulty_bucket(ep: Episode, *, edges: Sequence[float] = (0.0, 1.0, 3.0)) -> str:
    """Label easy/medium/hard from humans_near_path thresholds."""
    v = _difficulty_key(ep)
    e = list(edges)
    if v <= float(e[0]):
        return "easy"
    if len(e) >= 2 and v <= float(e[1]):
        return "medium"
    return "hard"


def split_fingerprint(split: Dict[str, Any]) -> str:
    payload = json.dumps(
        {"train": split.get("train"), "val": split.get("val"), "seed": split.get("seed")},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
