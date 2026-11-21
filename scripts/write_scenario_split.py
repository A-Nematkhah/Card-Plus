#!/usr/bin/env python
"""Create/overwrite scenario_split.json for an existing CrowdNav dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from card_plus.dataset.rescore import load_dataset_episodes
from card_plus.dataset.split import SPLIT_FILENAME, ensure_dataset_split, split_fingerprint


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--val-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    root = Path(args.dataset)
    episodes = load_dataset_episodes(root, split="all")
    split = ensure_dataset_split(
        root,
        episodes,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
        force=bool(args.force),
    )
    print(
        f"Wrote {root / SPLIT_FILENAME}: "
        f"train={len(split['train'])} val={len(split['val'])} "
        f"fp={split_fingerprint(split)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
