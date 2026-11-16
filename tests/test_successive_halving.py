"""Unit tests for successive-halving ranking helpers (no SAC)."""

from card_plus.dataset.successive_halving import (
    SHCandidate,
    parse_rung_steps,
    rank_candidates,
    select_survivors,
    survivors_to_keep,
)


def test_parse_rung_steps_csv():
    assert parse_rung_steps("50000,200000") == [50000, 200000]


def test_survivors_three_to_two_to_one():
    assert survivors_to_keep(3) == 2
    assert survivors_to_keep(2) == 1
    assert survivors_to_keep(1) == 1
    assert survivors_to_keep(4) == 2


def test_select_survivors_prefers_success_then_composite():
    cands = [
        SHCandidate(slot=0, attempt=0, specific_code="a", composite=0.9, success_rate=0.1),
        SHCandidate(slot=1, attempt=0, specific_code="b", composite=0.7, success_rate=0.4),
        SHCandidate(slot=2, attempt=0, specific_code="c", composite=0.8, success_rate=0.4),
    ]
    kept = select_survivors(cands)
    assert [c.slot for c in kept] == [2, 1]


def test_rank_tie_breaks_earlier_slot():
    cands = [
        SHCandidate(slot=2, attempt=0, specific_code="c", composite=0.5, success_rate=0.3),
        SHCandidate(slot=0, attempt=0, specific_code="a", composite=0.5, success_rate=0.3),
    ]
    assert rank_candidates(cands)[0].slot == 0
