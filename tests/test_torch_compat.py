"""torch.load weights_only compat for torch 1.11 vs 2.6+."""

from __future__ import annotations

import torch

from code_generation.self_reflection.torch_compat import install_torch_load_compat, torch_load


def test_torch_load_accepts_weights_only_on_old_or_new_torch(tmp_path):
    path = tmp_path / "t.pt"
    torch.save({"w": torch.ones(2)}, path)
    obj = torch_load(path, map_location="cpu", weights_only=False)
    assert "w" in obj
    # Also via monkeypatched global torch.load
    install_torch_load_compat()
    obj2 = torch.load(str(path), map_location="cpu", weights_only=False)
    assert "w" in obj2
