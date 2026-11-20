
import numpy as np
from typing import Tuple, Dict

def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    # Compute previous goal distance from pre_obs (explicitly provided for delta progress)
    prev_robot_pos = self.pre_obs[0:2]
    prev_goal = self.pre_obs[4:6]
    prev_goal_dist = np.linalg.norm(prev_robot_pos - prev_goal)
    current_goal_dist = float(np.linalg.norm(obs[0:2] - obs[4:6]))

    # 1. Progress toward goal (dense shaping)
    progress = prev_goal_dist - current_goal_dist
    progress_reward = 2.0 * progress

    # 2. Terminal success reward (large positive when goal reached without collision)
    success = current_goal_dist <= obs[6]
    success_reward = 100.0 if success else 0.0

    # 3. Terminal collision penalty (large negative)
    collision = obs[7] <= 0.0
    collision_penalty = -100.0 if collision else 0.0

    # 4. Comfort / clearance term (penalize getting closer than 0.5 m)
    clearance_threshold = 0.5
    clearance_reward = -max(0.0, clearance_threshold - obs[7]) * 10.0

    # 5. Freeze penalty (avoid stopping far from goal)
    if current_goal_dist > obs[6] + 0.5 and float(np.linalg.norm(obs[2:4])) < 0.05:
        freeze_penalty = -1.0
    else:
        freeze_penalty = 0.0

    # 6. Action regularization (small penalty on large actions)
    action_reg = -0.05 * np.sum(action ** 2)

    # Total reward
    reward = (progress_reward + success_reward + collision_penalty +
              clearance_reward + freeze_penalty + action_reg)

    # Dictionary of individual components
    reward_info = {
        'progress_reward': progress_reward,
        'success_reward': success_reward,
        'collision_penalty': collision_penalty,
        'clearance_reward': clearance_reward,
        'freeze_penalty': freeze_penalty,
        'action_reg': action_reg
    }

    return reward, reward_info
