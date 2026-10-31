"""Loop-boundary helpers: last train step is the real number of train rounds."""

from __future__ import annotations

from typing import Optional

from card_plus.config import CardPlusConfig


def is_final_train_step(step: int, max_step: Optional[int]) -> bool:
    """True iff this train step is the last one and no Coder query follows."""
    if max_step is None:
        return False
    return int(step) >= int(max_step)


def another_coder_query_follows(step: int, max_step: Optional[int]) -> bool:
    """Rollback is only allowed when another Coder query will run after this step."""
    if max_step is None:
        # Unknown horizon: do not treat as last (rollback may still fire).
        return True
    return int(step) < int(max_step)


def final_step_from_cfg(step: int, cfg: CardPlusConfig) -> bool:
    return is_final_train_step(step, cfg.card_plus_max_step)
