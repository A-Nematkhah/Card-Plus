"""Phase-3 helpers: running scale normalization + potential-based progress."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np

# Snippet injected into Coder prompts when structure/coeff separation is on.
STRUCTURE_ONLY_PROMPT = (
    "Write the reward STRUCTURE only: named sub-reward terms and how they combine. "
    "Use simple placeholder coefficients like 1.0 / 0.1 / 10.0; an offline optimizer "
    "will tune the numeric coefficients afterward. Prefer potential-based progress "
    "(reward ≈ Φ(pre) - Φ(post) with Φ = -goal_distance) so policy invariance is preserved. "
    "Do not spend tokens fine-tuning exact magnitudes."
)

POTENTIAL_PROGRESS_HELPER = '''
def potential_goal_phi(obs):
    """Φ(s) = -||pos - goal|| (higher when closer)."""
    import numpy as np
    return -float(np.linalg.norm(obs[0:2] - obs[4:6]))


def potential_based_progress(self, action, obs):
    """Ng-style F = γ Φ(s') - Φ(s); here γ=1, s'=obs, s=pre_obs."""
    del action
    import numpy as np
    phi_next = potential_goal_phi(obs)
    pre = getattr(self, "pre_obs", None)
    if pre is None:
        return 0.0
    phi_prev = potential_goal_phi(pre)
    return float(phi_next - phi_prev)
'''.strip()


@dataclass
class RunningScaleNormalizer:
    """Online mean/std normalizer for named sub-reward components."""

    eps: float = 1e-6
    clip: float = 5.0
    counts: Dict[str, int] = field(default_factory=dict)
    means: Dict[str, float] = field(default_factory=dict)
    m2: Dict[str, float] = field(default_factory=dict)

    def update(self, components: Mapping[str, float]) -> None:
        for k, v in components.items():
            x = float(v)
            n = int(self.counts.get(k, 0)) + 1
            mean = float(self.means.get(k, 0.0))
            delta = x - mean
            mean = mean + delta / n
            delta2 = x - mean
            self.m2[k] = float(self.m2.get(k, 0.0)) + delta * delta2
            self.counts[k] = n
            self.means[k] = mean

    def std(self, key: str) -> float:
        n = int(self.counts.get(key, 0))
        if n < 2:
            return 1.0
        var = float(self.m2.get(key, 0.0)) / float(n - 1)
        return float(max(var, 0.0) ** 0.5)

    def normalize(self, components: Mapping[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in components.items():
            mu = float(self.means.get(k, 0.0))
            sd = max(self.std(k), float(self.eps))
            z = (float(v) - mu) / sd
            if self.clip is not None:
                z = float(np.clip(z, -float(self.clip), float(self.clip)))
            out[str(k)] = z
        return out

    def normalize_and_update(self, components: Mapping[str, float]) -> Dict[str, float]:
        self.update(components)
        return self.normalize(components)


def potential_based_progress_value(
    pre_obs: Optional[np.ndarray],
    obs: np.ndarray,
) -> float:
    """Φ(s') - Φ(s) with Φ = -||pos-goal||."""
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    phi_next = -float(np.linalg.norm(obs[0:2] - obs[4:6]))
    if pre_obs is None:
        return 0.0
    pre = np.asarray(pre_obs, dtype=np.float64).reshape(-1)
    phi_prev = -float(np.linalg.norm(pre[0:2] - pre[4:6]))
    return float(phi_next - phi_prev)


def wrap_components_with_running_norm(
    components: MutableMapping[str, float],
    normalizer: RunningScaleNormalizer,
    *,
    keys: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Normalize selected (or all) component values."""
    if keys is None:
        subset = dict(components)
    else:
        subset = {k: float(components[k]) for k in keys if k in components}
    return normalizer.normalize_and_update(subset)
