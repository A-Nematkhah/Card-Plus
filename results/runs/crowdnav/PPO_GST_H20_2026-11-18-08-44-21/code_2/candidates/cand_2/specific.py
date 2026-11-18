
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    # New distinct structure with asymmetric progress weighting (stronger multiplier on negative deltas) + large overriding terminals for high outcome separation
    prev_goal_dist = np.linalg.norm(self.pre_obs[0:2] - self.pre_obs[4:6])
    progress = prev_goal_dist - float(np.linalg.norm(obs[0:2] - obs[4:6]))
    collision = obs[7] < 0.0
    success = (float(np.linalg.norm(obs[0:2] - obs[4:6])) <= obs[6]) and (obs[7] >= 0.0)
    reward_components: Dict[str, float] = {}
    if collision:
        total_reward = -400.0
        reward_components['collision'] = -400.0
    elif success:
        total_reward = 400.0
        reward_components['success'] = 400.0
    else:
        if progress > 0:
            progress_reward = progress * 7.0
        else:
            progress_reward = progress * 10.0
        total_reward = progress_reward
        reward_components['progress'] = progress_reward
    return total_reward, reward_components
