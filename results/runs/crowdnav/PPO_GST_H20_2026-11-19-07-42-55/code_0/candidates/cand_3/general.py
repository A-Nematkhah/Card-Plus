
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    # Asymmetric progress (high positive weight, low negative weight) for better maneuverability
    prev_goal_dist = np.linalg.norm(self.pre_obs[0:2] - self.pre_obs[4:6])
    delta_dist = prev_goal_dist - self.robot.goal_distance
    progress_reward = 3.2 * max(delta_dist, 0.0) + 0.15 * delta_dist

    # Amplified terminal rewards for stronger separation
    collision_reward = -30.0 if self.min_separation < 0.0 else 0.0
    success_reward = 30.0 if self.robot.goal_distance <= self.robot.radius and self.min_separation >= 0.0 else 0.0

    # Clearance penalty only when too close (no farming)
    clearance = self.min_separation
    clearance_reward = -10.0 * max(0.0, 0.3 - clearance) if clearance >= 0.0 else 0.0

    # Minimal action regularization
    action_penalty = -0.02 * np.dot(action, action)

    total_reward = progress_reward + collision_reward + success_reward + clearance_reward + action_penalty

    reward_dict = {
        'progress_reward': progress_reward,
        'collision_reward': collision_reward,
        'success_reward': success_reward,
        'clearance_reward': clearance_reward,
        'action_penalty': action_penalty
    }
    return total_reward, reward_dict
