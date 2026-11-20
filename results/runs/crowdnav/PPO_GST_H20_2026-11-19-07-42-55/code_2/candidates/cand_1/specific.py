
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    components = {}
    # Distinct structure with new weighting: progress delta only when positive (np.maximum(0, progress)) scaled to 5.0 for stronger goal progress signal; comfort penalty at -5.0 below 0.6m to tighten clearance without positive farming reward; collision at -12.0 and success at +45.0 to enlarge outcome gaps and boost anti_hack margin
    if self.pre_obs is not None:
        prev_goal_dist = np.linalg.norm(self.pre_obs[0:2] - self.pre_obs[4:6])
        progress = prev_goal_dist - float(np.linalg.norm(obs[0:2] - obs[4:6]))
    else:
        progress = 0.0
    progress_r = 5.0 * np.maximum(0.0, progress)
    components['progress'] = progress_r
    if obs[7] < 0.6:
        comfort_r = -5.0 * (0.6 - obs[7])
    else:
        comfort_r = 0.0
    components['comfort'] = comfort_r
    collision_r = -12.0 if obs[7] < 0 else 0.0
    components['collision'] = collision_r
    success_r = 45.0 if float(np.linalg.norm(obs[0:2] - obs[4:6])) <= obs[6] else 0.0
    components['success'] = success_r
    action_r = -0.1 * np.sum(action ** 2)
    components['action'] = action_r
    freeze_r = 0.0
    components['freeze'] = freeze_r
    total_reward = progress_r + comfort_r + collision_r + success_r + action_r + freeze_r
    return total_reward, components
