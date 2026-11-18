
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    # Different weighting: progress multiplier raised to 6.0 to further amplify penalty on negative deltas (improving anti_hack margin past 0.65) with adjusted large terminals for continued separation
    prev_goal_dist = np.linalg.norm(self.pre_obs[0:2] - self.pre_obs[4:6])
    progress = prev_goal_dist - self.robot.goal_distance
    collision = self.min_separation < 0.0
    success = (self.robot.goal_distance <= self.robot.radius) and (self.min_separation >= 0.0)
    reward_components: Dict[str, float] = {}
    if collision:
        total_reward = -250.0
        reward_components['collision'] = -250.0
    elif success:
        total_reward = 250.0
        reward_components['success'] = 250.0
    else:
        progress_reward = progress * 6.0
        total_reward = progress_reward
        reward_components['progress'] = progress_reward
    return total_reward, reward_components
