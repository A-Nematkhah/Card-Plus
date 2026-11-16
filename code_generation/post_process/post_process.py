import re
from typing import Dict

# Optional legacy LangChain imports (unused by RewardFunctionConverter).
try:
    from langchain.chains import LLMChain  # noqa: F401
except ImportError:
    LLMChain = None  # type: ignore
try:
    from langchain.base_language import BaseLanguageModel  # noqa: F401
except ImportError:
    BaseLanguageModel = None  # type: ignore


def _safe_token_replace(text: str, old: str, new: str) -> str:
    """Replace ``old`` with ``new`` only on identifier-like token boundaries."""
    if not old:
        return text
    # If the token is a plain identifier, use word boundaries; otherwise fall back
    # to literal replace (e.g. dotted paths like self.robot.position).
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", old):
        return re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text.replace(old, new)


class RewardFunctionConverter:
    def __init__(self, map_dict=None) -> None:
        self.map_dict = {k: map_dict[k] for k in
                         sorted(map_dict, key=len, reverse=True)} if map_dict is not None else {}

    def specific_to_general(self, specific_code: str) -> str:
        general_code = specific_code

        # replace specific terms with general terms
        for k, v in self.map_dict.items():
            general_code = _safe_token_replace(general_code, v, k)

        return general_code

    def general_to_specific(self, general_code: str) -> str:

        specific_code = general_code

        # replace general terms with specific terms
        for k, v in self.map_dict.items():
            specific_code = _safe_token_replace(specific_code, k, v)
        return specific_code