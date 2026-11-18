
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    # Step 1: Compute progress toward goal using previous observation
    prev_pos = self.pre_obs[0:2]
    prev_goal = self.pre_obs[4:6]
    prev_goal_dist = np.linalg.norm(prev_pos - prev_goal)
    progress = prev_goal_dist - self.robot.goal_distance

    # Step 2: Determine terminal conditions (collision or success)
    collision = self.min_separation < 0.0
    success = (self.robot.goal_distance <= self.robot.radius) and (self.min_separation >= 0.0)

    reward_components: Dict[str, float] = {}

    if collision:
        # Large negative terminal for collision to strongly discourage it
        total_reward = -100.0
        reward_components['collision'] = -100.0
    elif success:
        # Large positive terminal for success
        total_reward = 100.0
        reward_components['success'] = 100.0
    else:
        # New distinct structure: combine progress with a dense clearance penalty
        # This addresses the previous issue where collision dominated (72% reward mass)
        # and negative correlation by adding a balancing safety shaping term
        # Reason for change: pure progress led to fleeing/overshooting; clearance penalty
        # encourages keeping comfortable distance (>1m) while still rewarding progress,
        # improving sample efficiency without weakening terminal signals
        progress_reward = progress * 4.0
        # Clearance penalty: penalize low but non-colliding separation to promote comfort
        # and reduce risky close approaches that lead to timeout or collision
        if self.min_separation < 1.0:
            clearance_penalty = -3.0 * (1.0 - self.min_separation)
        else:
            clearance_penalty = 0.0
        total_reward = progress_reward + clearance_penalty
        reward_components['progress'] = progress_reward
        reward_components['clearance'] = clearance_penalty

    return total_reward, reward_components
