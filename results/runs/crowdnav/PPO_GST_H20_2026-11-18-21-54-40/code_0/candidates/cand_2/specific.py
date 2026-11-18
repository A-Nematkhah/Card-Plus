
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    reward_components = {}
    # Delta progress toward goal (positive when distance decreases)
    current_dist = float(np.linalg.norm(obs[0:2] - obs[4:6]))
    prev_dist = np.linalg.norm(self.pre_obs[0:2] - self.pre_obs[4:6])
    progress = prev_dist - current_dist
    reward_components['progress'] = progress
    # Large terminal success bonus when inside goal radius
    success = 50.0 if current_dist <= obs[6] else 0.0
    reward_components['success'] = success
    # Large terminal collision penalty
    collision = -50.0 if obs[7] < 0.0 else 0.0
    reward_components['collision'] = collision
    # Comfort penalty scaled by how far below safe clearance
    comfort = 0.0
    if obs[7] < 0.5:
        comfort = -3.0 * (0.5 - obs[7])
    reward_components['comfort'] = comfort
    # Quadratic action regularization (distinct from linear norm)
    action_reg = -0.08 * np.sum(np.square(action))
    reward_components['action_reg'] = action_reg
    # Anti-freeze penalty when nearly stopped far from goal
    freeze = -1.5 if (float(np.linalg.norm(obs[2:4])) < 0.05 and current_dist > 2.0 * obs[6]) else 0.0
    reward_components['freeze'] = freeze
    # Extra safety term using nearest human distance (distinct weighting)
    nearest_term = 0.0
    if obs[8] < 1.2:
        nearest_term = -0.8 * (1.2 - obs[8])
    reward_components['nearest_term'] = nearest_term
    total_reward = progress + success + collision + comfort + action_reg + freeze + nearest_term
    return total_reward, reward_components
