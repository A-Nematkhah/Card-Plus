
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    # Distinct structure: linear progress with high weight, tiered clearance (positive above threshold, penalty below), quadratic action cost, binary freeze check, and separate terminal bonuses; weights chosen to strongly prioritize goal reduction while enforcing clearance and motion.
    components = {}
    # Progress term: delta goal distance from pre_obs to encourage direct paths without detours
    if self.pre_obs is not None:
        prev_goal_dist = np.linalg.norm(self.pre_obs[0:2] - self.pre_obs[4:6])
        progress = prev_goal_dist - float(np.linalg.norm(obs[0:2] - obs[4:6]))
    else:
        progress = 0.0
    progress_r = 2.5 * progress
    components['progress'] = progress_r
    # Collision term: large negative on contact (min_separation < 0)
    collision_r = -15.0 if obs[7] < 0 else 0.0
    components['collision'] = collision_r
    # Comfort / clearance term: small positive reward when comfortably separated, linear penalty when too close
    if obs[7] >= 0.8:
        comfort_r = 0.2 * min(obs[7], 3.0)
    elif obs[7] < 0.3:
        comfort_r = -4.0 * (0.3 - obs[7])
    else:
        comfort_r = 0.0
    components['comfort'] = comfort_r
    # Action regularization: quadratic penalty on action magnitude to avoid excessive speeds
    action_r = -0.08 * np.sum(action ** 2)
    components['action'] = action_r
    # Anti-freeze term: penalize near-zero speed when still far from goal
    freeze_r = -1.2 if (float(np.linalg.norm(obs[2:4])) < 0.08 and float(np.linalg.norm(obs[0:2] - obs[4:6])) > 1.5) else 0.0
    components['freeze'] = freeze_r
    # Terminal success bonus (reaches within radius)
    success_r = 15.0 if float(np.linalg.norm(obs[0:2] - obs[4:6])) <= obs[6] else 0.0
    components['success'] = success_r
    total_reward = progress_r + collision_r + comfort_r + action_r + freeze_r + success_r
    return total_reward, components
