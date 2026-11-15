"""Unit tests for multi-candidate reward selection (no LLM)."""

from card_plus.dataset.candidate_loop import CandidateRecord, pick_best_accepted


def _rec(slot: int, attempt: int, composite: float, accepted: bool = True) -> CandidateRecord:
    class _Gate:
        def __init__(self, c: float) -> None:
            self.composite = c
            self.metrics = {"composite": c}
            self.reasons = []

    return CandidateRecord(
        slot=slot,
        attempt=attempt,
        general_code="g",
        specific_code="s",
        response=object(),
        gate=_Gate(composite),
        accepted=accepted,
    )


def test_pick_best_accepted_by_composite():
    accepted = [
        _rec(0, 0, 0.80),
        _rec(1, 2, 0.91),
        _rec(2, 0, 0.88),
    ]
    winner = pick_best_accepted(accepted)
    assert winner.slot == 1
    assert winner.attempt == 2
    assert winner.composite == 0.91


def test_pick_best_accepted_tie_prefers_earlier_slot():
    accepted = [_rec(0, 1, 0.90), _rec(2, 0, 0.90)]
    winner = pick_best_accepted(accepted)
    assert winner.slot == 0
