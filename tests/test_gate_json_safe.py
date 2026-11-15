"""Gate helpers: JSON-safe dumps and static short-circuit."""

from card_plus.dataset.gate import DatasetGateConfig, _json_safe, run_dataset_gate


def test_json_safe_replaces_nan_inf():
    payload = {"a": float("nan"), "b": float("inf"), "c": {"d": -float("inf")}, "e": 1.5}
    out = _json_safe(payload)
    assert out["a"] is None
    assert out["b"] is None
    assert out["c"]["d"] is None
    assert out["e"] == 1.5


def test_static_fail_short_circuits_without_dataset(tmp_path):
    # Empty dataset path would fail; create a fake dir so enabled=True then static fails first.
    ds = tmp_path / "ds"
    ds.mkdir()
    cfg = DatasetGateConfig(
        use_dataset_rescore=True,
        dataset_path=str(ds),
        dataset_mode="gate",
    )
    farm = """
def compute_dense_reward(self, action, obs):
    r_clearance = 0.1 * obs[7]
    return r_clearance, {"clearance": r_clearance}
"""
    result = run_dataset_gate(cfg, reward_code=farm, run_dir=str(tmp_path), step=0)
    assert result is not None
    assert result.keep is False
    assert any("static:" in r for r in result.reasons)
    assert result.composite == 0.0
