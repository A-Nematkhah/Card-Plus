"""Reward zoo construction for Phase-2 calibration (offline + optional online)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]


@dataclass
class ZooReward:
    reward_id: str
    code: str
    source: str = ""
    online_success: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward_id": self.reward_id,
            "code": self.code,
            "source": self.source,
            "online_success": self.online_success,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ZooReward":
        return cls(
            reward_id=str(d["reward_id"]),
            code=str(d["code"]),
            source=str(d.get("source") or ""),
            online_success=(
                None
                if d.get("online_success") is None
                else float(d["online_success"])
            ),
            meta=dict(d.get("meta") or {}),
        )


def _scale_numeric_constants(code: str, factor: float) -> str:
    """Scale decimal numeric literals (skip integers 0/1) via regex (Py3.8-safe)."""

    def _repl(m: re.Match) -> str:
        raw = m.group(0)
        try:
            v = float(raw)
        except ValueError:
            return raw
        if abs(v) in (0.0, 1.0):
            return raw
        return repr(v * float(factor))

    # Match floats / decimals; leave bare ints like 0,1 alone for flags.
    return re.sub(r"(?<![\w.])\d+\.\d+(?:[eE][+-]?\d+)?(?![\w.])", _repl, code)


def _drop_component_assignment(code: str, component: str) -> str:
    """Best-effort: zero dict value for ``component`` key in returned dict literal."""
    # Simple textual: "component": expr → "component": 0.0
    pattern = re.compile(
        rf'(["\']{re.escape(component)}["\']\s*:\s*)([^,}}]+)',
        re.MULTILINE,
    )
    if not pattern.search(code):
        return code
    return pattern.sub(rf"\g<1>0.0", code, count=1)


def mutate_reward_variants(base_code: str, base_id: str = "base") -> List[ZooReward]:
    """Hand mutations: scale coeffs, drop a term, inject a harmful clearance farm."""
    out: List[ZooReward] = [
        ZooReward(reward_id=base_id, code=base_code, source="base"),
    ]
    try:
        scaled_up = _scale_numeric_constants(base_code, 2.0)
        scaled_dn = _scale_numeric_constants(base_code, 0.5)
        if scaled_up != base_code:
            out.append(
                ZooReward(
                    reward_id=f"{base_id}__scale2",
                    code=scaled_up,
                    source="mutation_scale2",
                )
            )
        if scaled_dn != base_code:
            out.append(
                ZooReward(
                    reward_id=f"{base_id}__scale05",
                    code=scaled_dn,
                    source="mutation_scale05",
                )
            )
    except SyntaxError:
        pass

    for comp in ("progress", "clearance", "collision", "stall", "comfort"):
        dropped = _drop_component_assignment(base_code, comp)
        if dropped != base_code:
            out.append(
                ZooReward(
                    reward_id=f"{base_id}__drop_{comp}",
                    code=dropped,
                    source=f"mutation_drop_{comp}",
                )
            )

    harmful = '''
def compute_dense_reward(self, action, obs):
    import numpy as np
    r_progress = -0.01 * float(np.linalg.norm(obs[0:2] - obs[4:6]))
    r_clearance = 0.2 * float(obs[7])  # harmful farm
    total = r_progress + r_clearance
    if obs[7] < 0.0:
        total += -5.0
    elif float(np.linalg.norm(obs[0:2] - obs[4:6])) <= obs[6]:
        total += 10.0
    return total, {"progress": r_progress, "clearance": r_clearance}
'''
    out.append(
        ZooReward(
            reward_id=f"{base_id}__harmful_clearance",
            code=harmful.strip() + "\n",
            source="mutation_harmful",
        )
    )
    return out


def harvest_candidates_from_run(run_dir: PathLike) -> List[ZooReward]:
    """Collect specific.py from code_*/candidates/cand_* and gate keep/reject meta."""
    root = Path(run_dir)
    out: List[ZooReward] = []
    for cand in sorted(root.glob("code_*/candidates/cand_*")):
        specific = cand / "specific.py"
        if not specific.is_file():
            continue
        code = specific.read_text(encoding="utf-8")
        meta_path = cand / "meta.json"
        meta: Dict[str, Any] = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        rid = f"{cand.parent.parent.name}_{cand.name}"
        out.append(
            ZooReward(
                reward_id=rid,
                code=code,
                source="llm_candidate",
                meta=meta,
            )
        )
    return out


def load_zoo_json(path: PathLike) -> List[ZooReward]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data.get("rewards") if isinstance(data, dict) else data
    return [ZooReward.from_dict(r) for r in rows]


def save_zoo_json(rewards: Sequence[ZooReward], path: PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"n": len(rewards), "rewards": [r.to_dict() for r in rewards]}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def build_zoo(
    *,
    base_codes: Optional[Sequence[Tuple[str, str]]] = None,
    run_dirs: Optional[Sequence[PathLike]] = None,
    existing_zoo: Optional[PathLike] = None,
    include_mutations: bool = True,
) -> List[ZooReward]:
    """Assemble a zoo from base codes, run harvests, and optional prior zoo file."""
    by_id: Dict[str, ZooReward] = {}
    if existing_zoo is not None and Path(existing_zoo).is_file():
        for r in load_zoo_json(existing_zoo):
            by_id[r.reward_id] = r
    if run_dirs:
        for rd in run_dirs:
            for r in harvest_candidates_from_run(rd):
                by_id[r.reward_id] = r
    if base_codes:
        for rid, code in base_codes:
            by_id[rid] = ZooReward(reward_id=rid, code=code, source="base")
            if include_mutations:
                for m in mutate_reward_variants(code, base_id=rid):
                    by_id[m.reward_id] = m
    return list(by_id.values())
