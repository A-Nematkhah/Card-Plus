
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    # Distinct structure from prior versions: introduce small per-step time penalty (-0.1) on non-terminal steps to encourage shorter efficient paths and improve sample efficiency, while using scaled progress (x7.0) and larger terminal magnitudes for balance
    prev_goal_dist = np.linalg.norm(self.pre_obs[0:2] - self.pre_obs[4:6])
    progress = prev_goal_dist - float(np.linalg.norm(obs[0:2] - obs[4:6]))
    collision = obs[7] < 0.0
    success = (float(np.linalg.norm(obs[0:2] - obs[4:6])) <= obs[6]) and (obs[7] >= 0.0)
    reward_components: Dict[str, float] = {}
    if collision:
        total_reward = -300.0
        reward_components['collision'] = -300.0
    elif success:
        total_reward = 300.0
        reward_components['success'] = 300.0
    else:
        progress_reward = progress * 7.0
        time_cost = -0.1
        total_reward = progress_reward + time_cost
        reward_components['progress'] = progress_reward
        reward_components['time_cost'] = time_cost
    return total_reward, reward_components
