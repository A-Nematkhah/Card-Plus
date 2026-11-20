
import numpy as np
from typing import Tuple, Dict

def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    pre_pos = self.pre_obs[0:2]
    pre_goal = self.pre_obs[4:6]
    pre_dist = np.linalg.norm(pre_pos - pre_goal)
    curr_dist = self.robot.goal_distance
    # High-weighted progress to ensure positive mean on approach probes
    progress_reward = 12.0 * (pre_dist - curr_dist)

    # Per-step time cost (distinct structure) to penalize loitering/orbit without affecting fast progress
    time_penalty = -0.08

    is_collision = self.min_separation <= 0.0
    is_success = curr_dist <= self.robot.radius
    collision_penalty = -300.0 if is_collision else 0.0
    success_reward = 400.0 if is_success else 0.0

    # Clearance penalty strictly when below threshold; no positive term or bonus for high values
    clearance = self.min_separation
    clearance_penalty = -6.0 * max(0.0, 1.0 - clearance) if clearance < 1.0 else 0.0

    action_norm = np.linalg.norm(action)
    action_reg = -0.005 * action_norm

    # Freeze only when stalled far away
    freeze_penalty = -1.5 if (self.robot.speed < 0.05 and curr_dist > 0.8) else 0.0

    total_reward = progress_reward + time_penalty + collision_penalty + success_reward + clearance_penalty + action_reg + freeze_penalty

    reward_dict = {
        'progress': progress_reward,
        'time': time_penalty,
        'collision': collision_penalty,
        'success': success_reward,
        'clearance': clearance_penalty,
        'action_reg': action_reg,
        'freeze': freeze_penalty
    }
    return total_reward, reward_dict
