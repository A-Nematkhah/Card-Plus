
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    # Distinct structure: introduce small continuous comfort penalty based on min_separation in non-terminal case (different from pure progress+terminals); scaled terminals and progress to reduce collision dominance while keeping dense shaping
    prev_goal_dist = np.linalg.norm(self.pre_obs[0:2] - self.pre_obs[4:6])
    progress = prev_goal_dist - float(np.linalg.norm(obs[0:2] - obs[4:6]))
    collision = obs[7] < 0.0
    success = (float(np.linalg.norm(obs[0:2] - obs[4:6])) <= obs[6]) and (obs[7] >= 0.0)
    reward_components: Dict[str, float] = {}
    if success:
        total_reward = 100.0
        reward_components['success'] = 100.0
    elif collision:
        total_reward = -100.0
        reward_components['collision'] = -100.0
    else:
        progress_reward = progress * 4.0
        comfort = max(0.0, 1.0 - obs[7]) * -0.5
        total_reward = progress_reward + comfort
        reward_components['progress'] = progress_reward
        reward_components['comfort'] = comfort
    return total_reward, reward_components
