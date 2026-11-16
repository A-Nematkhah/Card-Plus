"""Legacy successive-halving entrypoint (disabled).

Per-candidate short RL rungs were removed from the default CrowdNav CARD loop.
Gate-accepted candidates are ranked by composite at query time; long-budget
selection of archive top-K is handled by ``crowdnav_finalists``.

This module stays importable so older scripts do not crash; enabling
``--use_successive_halving`` now keeps the composite winner and prints a notice.
"""

from __future__ import annotations

from typing import Any, Optional

from code_generation.self_reflection.utils import print_with_color


def maybe_select_reward_via_sh(
    args: Any,
    *,
    run_dir: str,
    code_step: int,
    specific_code_path: str,
    general_code_path: str,
) -> Optional[str]:
    """No-op: per-candidate SH rungs are gone. Prefer ``crowdnav_finalists``."""
    _ = (run_dir, code_step, specific_code_path, general_code_path)
    if not bool(getattr(args, "use_successive_halving", False)):
        return None
    print_with_color(
        "SH: per-candidate rungs removed from the default CrowdNav loop; "
        "keeping the composite gate winner. Use crowdnav_finalists for "
        "long-budget top-K selection.",
        "YELLOW",
    )
    return None
