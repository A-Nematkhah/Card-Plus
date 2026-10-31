# Task instructions and observation mappings used by CARD CrowdNav++.

from code_generation.self_reflection.crowdnav_env_utils import CROWDNAV_TASKS, mapping_dicts

instruction_mapping = {name: meta["instruction"] for name, meta in CROWDNAV_TASKS.items()}

__all__ = ["instruction_mapping", "mapping_dicts"]
