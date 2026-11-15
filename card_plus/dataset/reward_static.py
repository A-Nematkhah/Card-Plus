"""Static checks that reject known reward-hacking patterns in LLM reward code.

Main rule: never pay a positive dense bonus for larger min_separation / obs[7]
(clearance farming / fleeing). Clearance must be penalty-only when close.
"""

from __future__ import annotations

import ast
import re
from typing import List, Tuple

# Flat CARD obs index for min_separation (see crowdnav_env_utils.mapping_dicts).
_OBS_SEP_INDEX = 7


def _is_obs_sep(node: ast.AST) -> bool:
    """True if node is ``obs[7]`` (or obs[7:8] style not used)."""
    if not isinstance(node, ast.Subscript):
        return False
    if not isinstance(node.value, ast.Name) or node.value.id != "obs":
        return False
    sl = node.slice
    # py3.9+: slice is Constant/Index-like; py3.8 used Index
    if isinstance(sl, ast.Constant) and sl.value == _OBS_SEP_INDEX:
        return True
    if isinstance(sl, ast.UnaryOp) and isinstance(sl.op, ast.UAdd):
        return isinstance(sl.operand, ast.Constant) and sl.operand.value == _OBS_SEP_INDEX
    return False


def _is_min_separation_attr(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "min_separation"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _is_sep_expr(node: ast.AST) -> bool:
    return _is_obs_sep(node) or _is_min_separation_attr(node)


def _positive_number(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value) > 0.0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _positive_number(node.operand)
    return False


def _negated(node: ast.AST) -> bool:
    return isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)


def _mult_positive_sep_bonus(node: ast.AST) -> bool:
    """True for ``(+c) * sep`` or ``sep * (+c)`` not wrapped in unary minus."""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
        return False
    left, right = node.left, node.right
    if _is_sep_expr(left) and _positive_number(right):
        return True
    if _is_sep_expr(right) and _positive_number(left):
        return True
    return False


def _walk_reject_positive_sep(tree: ast.AST) -> List[str]:
    reasons: List[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_BinOp(self, node: ast.BinOp) -> None:
            # Skip if this Mult is immediately negated: -0.1 * obs[7]
            # Parent check happens via visit_UnaryOp wrapping; here flag raw Mult.
            if _mult_positive_sep_bonus(node):
                reasons.append(
                    "positive dense bonus on min_separation/obs[7] "
                    "(clearance farming / flee hack); use penalty-only when close"
                )
            self.generic_visit(node)

        def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
            # Do not flag children of unary minus as positive bonus.
            if isinstance(node.op, ast.USub) and _mult_positive_sep_bonus(node.operand):
                return
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            if isinstance(node.op, ast.Add) and (
                _is_sep_expr(node.value) or _mult_positive_sep_bonus(node.value)
            ):
                reasons.append(
                    "augmented add of min_separation/obs[7] into reward "
                    "(clearance farming); clearance must be penalty-only"
                )
            self.generic_visit(node)

    Visitor().visit(tree)
    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


_CLEARANCE_POS_ASSIGN = re.compile(
    r"(?i)\b(r_)?clearance\b[^\n]{0,60}=\s*(?:\+)?\s*"
    r"((?:\d+\.?\d*)|(?:\.\d+))\s*\*\s*"
    r"(?:obs\s*\[\s*7\s*\]|self\.min_separation)"
)


def check_reward_code_static(code: str) -> Tuple[bool, List[str]]:
    """Return (ok, reasons). ok=False means reject / rewrite."""
    reasons: List[str] = []
    if not code or not str(code).strip():
        return False, ["empty reward code"]

    text = str(code)
    m = _CLEARANCE_POS_ASSIGN.search(text)
    if m and float(m.group(2)) > 0.0:
        reasons.append(
            f"clearance assigned as +{m.group(2)}*separation "
            "(forbidden; use penalty only when sep < comfort)"
        )

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return False, [f"syntax error: {exc}"]

    reasons.extend(_walk_reject_positive_sep(tree))
    return (len(reasons) == 0), reasons
