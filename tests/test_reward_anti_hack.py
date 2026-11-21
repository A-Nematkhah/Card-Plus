"""Static rules + synthetic anti-hack probes for CrowdNav rewards."""

from card_plus.dataset.anti_hack_probes import run_anti_hack_probes, run_suicide_probe
from card_plus.dataset.rescore import load_reward_fn_from_code
from card_plus.dataset.reward_static import check_reward_code_static

FARM_CODE = """
def compute_dense_reward(self, action, obs):
    r_progress = -0.02 * float(__import__('numpy').linalg.norm(obs[0:2] - obs[4:6]))
    r_clearance = 0.1 * obs[7]
    total = r_progress + r_clearance
    if obs[7] < 0.0:
        total += -50.0
    elif float(__import__('numpy').linalg.norm(obs[0:2] - obs[4:6])) <= obs[6]:
        total += 200.0
    return total, {"progress": r_progress, "clearance": r_clearance}
"""

PENALTY_ONLY_CODE = """
def compute_dense_reward(self, action, obs):
    import numpy as np
    cur_d = float(np.linalg.norm(obs[0:2] - obs[4:6]))
    pre = getattr(self, "pre_obs", None)
    if pre is None:
        prev_d = cur_d
    else:
        prev_d = float(np.linalg.norm(pre[0:2] - pre[4:6]))
    r_progress = 1.0 * (prev_d - cur_d)
    comfort = 0.5
    if obs[7] < comfort:
        r_clearance = -5.0 * (comfort - obs[7])
    else:
        r_clearance = 0.0
    total = r_progress + r_clearance
    if obs[7] < 0.0:
        total += -100.0
        return total, {"progress": r_progress, "clearance": r_clearance, "collision": -100.0}
    if cur_d <= obs[6]:
        total += 500.0
        return total, {"progress": r_progress, "clearance": r_clearance, "success": 500.0}
    return total, {"progress": r_progress, "clearance": r_clearance}
"""


def test_static_rejects_positive_clearance_farm():
    ok, reasons = check_reward_code_static(FARM_CODE)
    assert ok is False
    assert any("clearance" in r.lower() or "separation" in r.lower() or "obs[7]" in r for r in reasons)


def test_static_allows_penalty_only_clearance():
    ok, reasons = check_reward_code_static(PENALTY_ONLY_CODE)
    assert ok is True, reasons


def test_probes_reject_farm_reward():
    fn = load_reward_fn_from_code(FARM_CODE)
    result = run_anti_hack_probes(fn)
    assert result.ok is False
    assert any("flee" in r for r in result.reasons), result.reasons


def test_probes_accept_penalty_only_delta_progress():
    fn = load_reward_fn_from_code(PENALTY_ONLY_CODE)
    result = run_anti_hack_probes(fn)
    assert result.ok is True, result.reasons


SUICIDE_BAD_CODE = """
def compute_dense_reward(self, action, obs):
    # Tiny collision, huge per-step stall → early crash preferred over waiting.
    stall = -1.0
    collision = -2.0 if obs[7] < 0.0 else 0.0
    total = stall + collision
    return total, {"stall": stall, "collision": collision}
"""

SUICIDE_OK_CODE = """
def compute_dense_reward(self, action, obs):
    stall = -0.01
    collision = -50.0 if obs[7] < 0.0 else 0.0
    total = stall + collision
    return total, {"stall": stall, "collision": collision}
"""


def test_suicide_probe_rejects_early_crash_preference():
    fn = load_reward_fn_from_code(SUICIDE_BAD_CODE)
    result = run_suicide_probe(fn)
    assert result.ok is False, result.metrics
    assert any("suicide" in r for r in result.reasons)


def test_suicide_probe_accepts_large_collision_penalty():
    fn = load_reward_fn_from_code(SUICIDE_OK_CODE)
    result = run_suicide_probe(fn)
    assert result.ok is True, result.reasons


def test_anti_hack_optional_suicide_flag_off_by_default():
    fn = load_reward_fn_from_code(SUICIDE_BAD_CODE)
    # Without flag, flee/orbit probes may still fail farm-like rewards; use OK base
    # structure that only fails suicide.
    ok_base = load_reward_fn_from_code(PENALTY_ONLY_CODE)
    base = run_anti_hack_probes(ok_base, use_suicide_probe=False)
    assert base.ok is True
    with_suicide = run_anti_hack_probes(fn, use_suicide_probe=True)
    # SUICIDE_BAD may also fail flee/orbit; at least suicide metrics present.
    assert "suicide_mean" in with_suicide.metrics
