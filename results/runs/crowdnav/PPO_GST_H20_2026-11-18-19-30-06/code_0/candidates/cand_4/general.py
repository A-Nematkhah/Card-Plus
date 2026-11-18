
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    # Progress via distance reduction (use pre_obs for previous goal distance)
    prev_pos = self.pre_obs[0:2]
    prev_goal = self.pre_obs[4:6]
    prev_goal_dist = np.linalg.norm(prev_pos - prev_goal)
    curr_goal_dist = self.robot.goal_distance
    progress = prev_goal_dist - curr_goal_dist
    progress_reward = 0.8 * progress

    # Terminal conditions
    is_collision = self.min_separation <= 0.0
    is_success = (curr_goal_dist <= self.robot.radius) and (not is_collision)
    collision_reward = -20.0 if is_collision else 0.0
    success_reward = 20.0 if is_success else 0.0

    # Comfort / clearance using nearest three humans (stronger penalty when close)
    min_sep = self.min_separation
    comfort_reward = 0.0
    if min_sep < 0.6:
        comfort_reward -= 3.0 * (0.6 - min_sep)
    if self.second_nearest_human.distance < 0.9:
        comfort_reward -= 1.2 * (0.9 - self.second_nearest_human.distance)
    if self.third_nearest_human.distance < 0.9:
        comfort_reward -= 0.6 * (0.9 - self.third_nearest_human.distance)

    # Action regularization (penalize large controls)
    action_reg = -0.02 * np.linalg.norm(action)**2

    # Anti-freeze term (discourage stopping when far from goal)
    freeze_penalty = 0.0
    if self.robot.speed < 0.15 and curr_goal_dist > self.robot.radius + 0.2:
        freeze_penalty = -0.4

    # Combine with distinct weighting
    total_reward = (progress_reward + collision_reward + success_reward +
                    comfort_reward + action_reg + freeze_penalty)

    reward_dict = {
        'progress': progress_reward,
        'collision': collision_reward,
        'success': success_reward,
        'comfort': comfort_reward,
        'action_reg': action_reg,
        'freeze': freeze_penalty
    }
    return total_reward, reward_dict
