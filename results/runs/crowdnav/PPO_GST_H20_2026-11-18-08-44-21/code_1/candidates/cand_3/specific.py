
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    # Modified structure: removed positive clearance bonus (which enabled farming/flee); replaced with penalty-only when min_separation < 1.0 to avoid hack while encouraging safe distance.
    # Reason for change: static check failed on positive min_separation bonus; now penalty-only when close fixes farming, uses only listed attrs like float(np.linalg.norm(obs[0:2] - obs[4:6])) and obs[7].
    # Distinct weighting: progress*3.0, terminals at -80/+80 for balanced mass, clearance penalty scaled to -1.5*(1-sep) when violated.
    prev_pos = self.pre_obs[0:2]
    prev_goal = self.pre_obs[4:6]
    prev_goal_dist = np.linalg.norm(prev_pos - prev_goal)
    progress = prev_goal_dist - float(np.linalg.norm(obs[0:2] - obs[4:6]))

    collision = obs[7] < 0.0
    success = (float(np.linalg.norm(obs[0:2] - obs[4:6])) <= obs[6]) and (obs[7] >= 0.0)

    reward_components: Dict[str, float] = {}
    if collision:
        total_reward = -80.0
        reward_components['collision'] = -80.0
    elif success:
        total_reward = 80.0
        reward_components['success'] = 80.0
    else:
        progress_reward = progress * 3.0
        # penalty-only when close: negative term if min_separation below 1.0m, zero otherwise
        if obs[7] < 1.0:
            clearance_penalty = -1.5 * (1.0 - obs[7])
        else:
            clearance_penalty = 0.0
        total_reward = progress_reward + clearance_penalty
        reward_components['progress'] = progress_reward
        reward_components['clearance'] = clearance_penalty

    return total_reward, reward_components
