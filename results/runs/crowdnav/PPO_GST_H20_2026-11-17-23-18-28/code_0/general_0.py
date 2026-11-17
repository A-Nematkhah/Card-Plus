
import numpy as np
from typing import Tuple, Dict

def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    # Asymmetric goal progress (stronger for closing gap, milder for moving away)
    if self.pre_obs is not None:
        prev_dist = np.linalg.norm(self.pre_obs[0:2] - self.pre_obs[4:6])
        delta = prev_dist - self.robot.goal_distance
        if delta > 0.0:
            goal_progress = 5.0 * delta
        else:
            goal_progress = 1.5 * delta
    else:
        goal_progress = 0.0

    # Clearance: penalty only when close
    min_sep = self.min_separation
    if min_sep < 0.0:
        collision_term = -50.0
        close_penalty = 0.0
    elif min_sep < 0.3:
        close_penalty = -6.0 * (0.3 - min_sep)
        collision_term = 0.0
    else:
        close_penalty = 0.0
        collision_term = 0.0

    # Terminal success
    success_term = 35.0 if self.robot.goal_distance <= self.robot.radius else 0.0

    # Mild freeze penalty
    freeze_penalty = -0.3 if (self.robot.speed < 0.1 and self.robot.goal_distance > self.robot.radius) else 0.0

    # Action regularization
    action_reg = -0.01 * np.sum(action ** 2)

    reward = goal_progress + close_penalty + collision_term + success_term + freeze_penalty + action_reg

    components = {
        "goal_progress": goal_progress,
        "close_penalty": close_penalty,
        "collision_term": collision_term,
        "success_term": success_term,
        "freeze_penalty": freeze_penalty,
        "action_reg": action_reg
    }
    return reward, components
