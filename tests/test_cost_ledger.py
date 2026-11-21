"""Cost ledger + VersionRecord phase-0 fields."""

from __future__ import annotations

import json

from card_plus.archive import VersionRecord
from card_plus.logging_utils import append_cost_event


def test_append_cost_event(tmp_path):
    path = append_cost_event(
        str(tmp_path),
        stage="unit_test",
        iteration=0,
        prompt_tokens=11,
        completion_tokens=22,
        llm_calls=1,
        rl_steps=100,
        wall_s=1.5,
        extra={"ok": True},
    )
    assert path.name == "card_plus_cost.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["stage"] == "unit_test"
    assert row["prompt_tokens"] == 11
    assert row["completion_tokens"] == 22
    assert row["llm_calls"] == 1
    assert row["extra"]["ok"] is True
    assert "ts" in row


def test_version_record_gate_fields_roundtrip():
    rec = VersionRecord(
        iteration=3,
        reward_code="x",
        gate_composite=0.81,
        gate_metrics={"anti_hack": 0.7},
        online_success=0.92,
    )
    d = rec.to_dict()
    back = VersionRecord.from_dict(d)
    assert back.gate_composite == 0.81
    assert back.gate_metrics == {"anti_hack": 0.7}
    assert back.online_success == 0.92
    # Old archives without new keys still load.
    old = VersionRecord.from_dict({"iteration": 1, "reward_code": "y"})
    assert old.gate_composite is None
    assert old.online_success is None
