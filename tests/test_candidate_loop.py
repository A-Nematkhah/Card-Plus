"""Unit tests for multi-candidate reward selection (no LLM)."""

from card_plus.dataset.candidate_loop import (
    CandidateRecord,
    learnability_bonus,
    pick_best_accepted,
)


def _rec(
    slot: int,
    attempt: int,
    composite: float,
    accepted: bool = True,
    *,
    mean_success: float = 20.0,
    mean_collided: float = -25.0,
    mean_timeout: float = -5.0,
    return_std: float = 30.0,
) -> CandidateRecord:
    class _Gate:
        def __init__(self, c: float) -> None:
            self.composite = c
            self.metrics = {
                "composite": c,
                "mean_success": mean_success,
                "mean_collided": mean_collided,
                "mean_timeout": mean_timeout,
                "return_std": return_std,
            }
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


def test_pick_best_prefers_learnable_scale_on_near_tie():
    # Near-equal composite: huge collision scale should lose to mild terminals.
    harsh = _rec(
        0,
        0,
        0.887,
        mean_success=50.0,
        mean_collided=-980.0,
        mean_timeout=-570.0,
        return_std=340.0,
    )
    mild = _rec(
        1,
        0,
        0.880,
        mean_success=25.0,
        mean_collided=-30.0,
        mean_timeout=-8.0,
        return_std=35.0,
    )
    assert learnability_bonus(mild.gate) > learnability_bonus(harsh.gate)
    winner = pick_best_accepted([harsh, mild])
    assert winner.slot == 1
