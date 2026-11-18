"""Atomic PPO checkpoint save / corrupt detection."""

from pathlib import Path

import torch

from code_generation.self_reflection.crowdnav_ppo_train import (
    atomic_torch_save,
    is_usable_torch_checkpoint,
    read_checkpoint_meta,
    remove_corrupt_checkpoint,
)


def test_atomic_save_and_meta(tmp_path: Path):
    path = str(tmp_path / "latest_model.pt")
    atomic_torch_save({"w": torch.zeros(3)}, path, steps=12000, complete=False)
    assert is_usable_torch_checkpoint(path)
    meta = read_checkpoint_meta(path)
    assert meta["steps"] == 12000
    assert meta["complete"] is False
    atomic_torch_save({"w": torch.ones(3)}, path, steps=250000, complete=True)
    assert read_checkpoint_meta(path)["complete"] is True


def test_corrupt_checkpoint_detected(tmp_path: Path):
    path = str(tmp_path / "latest_model.pt")
    Path(path).write_bytes(b"not-a-real-pytorch-zip-file" + b"\0" * 128)
    assert not is_usable_torch_checkpoint(path)
    remove_corrupt_checkpoint(path)
    assert not Path(path).exists()
