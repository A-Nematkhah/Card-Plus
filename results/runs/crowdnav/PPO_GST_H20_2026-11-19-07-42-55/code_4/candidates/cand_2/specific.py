
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    components = {}
    # Distinct structure: progress weight raised to 10.0 with clipped positive delta only (stronger goal signal to drive successes); comfort penalty restricted to <0.3m with milder -2.0 scale (reduces negative corr and collision dominance); collision -15/success +15 for separation; action cost minimal.
    if self.pre_obs is not None:
        prev_goal_dist = np.linalg.norm(self.pre_obs[0:2] - self.pre_obs[4:6])
        progress = prev_goal_dist - float(np.linalg.norm(obs[0:2] - obs[4:6]))
    else:
        progress = 0.0
    progress_r = 10.0 * np.maximum(0.0, progress)
    components['progress'] = progress_r
    if obs[7] < 0.3:
        comfort_r = -2.0 * (0.3 - obs[7])
    else:
        comfort_r = 0.0
    components['comfort'] = comfort_r
    collision_r = -15.0 if obs[7] < 0 else 0.0
    components['collision'] = collision_r
    success_r = 15.0 if float(np.linalg.norm(obs[0:2] - obs[4:6])) <= obs[6] else 0.0
    components['success'] = success_r
    action_r = -0.02 * np.sum(action ** 2)
    components['action'] = action_r
    freeze_r = 0.0
    components['freeze'] = freeze_r
    total_reward = progress_r + comfort_r + collision_r + success_r + action_r + freeze_r
    return total_reward, components
