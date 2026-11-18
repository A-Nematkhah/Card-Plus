
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    prev_pos = self.pre_obs[0:2]
    prev_goal = self.pre_obs[4:6]
    prev_goal_dist = np.linalg.norm(prev_pos - prev_goal)
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
        progress_reward = progress * 7.0
        total_reward = progress_reward
        reward_components['progress'] = progress_reward

    return total_reward, reward_components
