"""Torch helpers that work on both old (1.11) and new (2.6+) PyTorch.

``weights_only`` was added in newer torch.load; CrowdNav GST wrappers pass it
for PyTorch 2.6+ safety, but the ``card`` env ships torch 1.11 where that kwarg
is forwarded into pickle.Unpickler and crashes:

    TypeError: 'weights_only' is an invalid keyword argument for Unpickler()

Install the monkeypatch once before any GST / PPO checkpoint load.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

_INSTALLED = False


def torch_load(
    f: Any,
    map_location: Any = None,
    *,
    weights_only: Optional[bool] = None,
    **kwargs: Any,
) -> Any:
    """``torch.load`` that ignores unsupported ``weights_only`` on old PyTorch."""
    kw = dict(kwargs)
    if map_location is not None:
        kw["map_location"] = map_location
    if weights_only is None:
        return torch.load(f, **kw)
    try:
        return torch.load(f, weights_only=weights_only, **kw)
    except TypeError as exc:
        msg = str(exc)
        if "weights_only" not in msg and "Unpickler" not in msg:
            raise
        return torch.load(f, **kw)


def install_torch_load_compat() -> None:
    """Make global ``torch.load`` tolerate ``weights_only`` (idempotent)."""
    global _INSTALLED
    if _INSTALLED or getattr(torch.load, "_card_weights_only_compat", False):
        _INSTALLED = True
        return

    _orig = torch.load

    def _compat_load(*args: Any, **kwargs: Any) -> Any:
        if "weights_only" not in kwargs:
            return _orig(*args, **kwargs)
        try:
            return _orig(*args, **kwargs)
        except TypeError as exc:
            msg = str(exc)
            if "weights_only" not in msg and "Unpickler" not in msg:
                raise
            kwargs = dict(kwargs)
            kwargs.pop("weights_only", None)
            return _orig(*args, **kwargs)

    _compat_load._card_weights_only_compat = True  # type: ignore[attr-defined]
    torch.load = _compat_load  # type: ignore[assignment]
    _INSTALLED = True
