
import numpy as np
from typing import Tuple, Dict

def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    # delta progress using pre_obs (distinct from prior versions by using L1 action reg + scaled freeze)
    if self.pre_obs is not None:
        prev_pos = self.pre_obs[0:2]
        prev_goal = self.pre_obs[4:6]
        prev_goal_dist = np.linalg.norm(prev_pos - prev_goal)
    else:
        prev_goal_dist = float(np.linalg.norm(obs[0:2] - obs[4:6]))
    goal_progress = prev_goal_dist - float(np.linalg.norm(obs[0:2] - obs[4:6]))
    progress_reward = 3.0 * goal_progress

    # clearance / collision (distinct weighting and threshold structure)
    min_sep = obs[7]
    if min_sep <= 0:
        collision_penalty = -15.0
        clearance_reward = -5.0
    else:
        collision_penalty = 0.0
        if min_sep >= 1.0:
            clearance_reward = 1.0
        else:
            clearance_reward = min_sep - 1.0

    # terminal success (distinct large value)
    success_reward = 0.0
    if float(np.linalg.norm(obs[0:2] - obs[4:6])) <= obs[6]:
        success_reward = 25.0

    # action regularization (distinct L1 norm)
    action_reg = -0.05 * np.sum(np.abs(action))

    # anti-freeze (distinct scaling)
    freeze_penalty = 0.0
    if float(np.linalg.norm(obs[2:4])) < 0.15 and float(np.linalg.norm(obs[0:2] - obs[4:6])) > obs[6] * 3:
        freeze_penalty = -0.3 * (1 - float(np.linalg.norm(obs[2:4])) / 0.15)

    total_reward = progress_reward + clearance_reward + collision_penalty + success_reward + action_reg + freeze_penalty
    reward_components = {
        "progress_reward": progress_reward,
        "clearance_reward": clearance_reward,
        "collision_penalty": collision_penalty,
        "success_reward": success_reward,
        "action_reg": action_reg,
        "freeze_penalty": freeze_penalty
    }
    return total_reward, reward_components
