"""Extract / inject numeric literals in ``compute_dense_reward`` via AST."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass
class CoeffSite:
    index: int
    value: float
    lineno: Optional[int] = None
    col_offset: Optional[int] = None


def _is_tunable_float(node: ast.AST, *, skip_unit: bool) -> Optional[float]:
    """Only float literals are tunable (ints are often slice indices / flags)."""
    if not isinstance(node, ast.Constant):
        return None
    if isinstance(node.value, bool) or node.value is None:
        return None
    # Prefer true floats; skip ints entirely (obs[0:2], counts, …).
    if not isinstance(node.value, float):
        return None
    value = float(node.value)
    if skip_unit and abs(value) in (0.0, 1.0):
        return None
    return value


def extract_coeffs(
    code: str,
    *,
    skip_unit_literals: bool = True,
) -> Tuple[List[float], List[CoeffSite]]:
    """Return (values, sites) for tunable float literals in ``ast.walk`` order."""
    tree = ast.parse(code)
    values: List[float] = []
    sites: List[CoeffSite] = []
    for node in ast.walk(tree):
        v = _is_tunable_float(node, skip_unit=skip_unit_literals)
        if v is None:
            continue
        idx = len(values)
        values.append(float(v))
        sites.append(
            CoeffSite(
                index=idx,
                value=float(v),
                lineno=getattr(node, "lineno", None),
                col_offset=getattr(node, "col_offset", None),
            )
        )
    return values, sites


def inject_coeffs(
    code: str,
    new_values: Sequence[float],
    *,
    skip_unit_literals: bool = True,
) -> str:
    """Return code with tunable float literals replaced in extract order."""
    tree = ast.parse(code)
    nodes: List[ast.AST] = []
    for node in ast.walk(tree):
        if _is_tunable_float(node, skip_unit=skip_unit_literals) is not None:
            nodes.append(node)
    if len(new_values) < len(nodes):
        # Pad with originals if short (should not happen in optimizer).
        originals, _ = extract_coeffs(code, skip_unit_literals=skip_unit_literals)
        vals = list(new_values) + originals[len(new_values) :]
    else:
        vals = list(new_values)
    for node, nv in zip(nodes, vals):
        if isinstance(node, ast.Constant):
            node.value = float(nv)
    ast.fix_missing_locations(tree)
    if hasattr(ast, "unparse"):
        return ast.unparse(tree) + "\n"
    return _inject_text_fallback(code, vals, skip_unit_literals=skip_unit_literals)


def _inject_text_fallback(
    code: str,
    new_values: Sequence[float],
    *,
    skip_unit_literals: bool,
) -> str:
    """Replace float tokens left-to-right (Py3.8 without ast.unparse)."""
    pattern = re.compile(
        r"(?<![\w.])[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])"
    )
    vals = list(new_values)
    i = 0

    def repl(m: re.Match) -> str:
        nonlocal i
        raw = m.group(0)
        try:
            v = float(raw)
        except ValueError:
            return raw
        if skip_unit_literals and abs(v) in (0.0, 1.0):
            return raw
        if i >= len(vals):
            return raw
        nv = vals[i]
        i += 1
        return repr(float(nv))

    return pattern.sub(repl, code)


def coeff_bounds(
    values: Sequence[float],
    *,
    scale_lo: float = 0.1,
    scale_hi: float = 10.0,
    abs_floor: float = 1e-3,
) -> List[Tuple[float, float]]:
    """Multiplicative bounds around each coefficient (sign-preserving)."""
    bounds: List[Tuple[float, float]] = []
    for v in values:
        mag = max(abs(float(v)), float(abs_floor))
        sign = 1.0 if v >= 0 else -1.0
        lo = sign * mag * float(scale_lo)
        hi = sign * mag * float(scale_hi)
        if lo > hi:
            lo, hi = hi, lo
        bounds.append((float(lo), float(hi)))
    return bounds
