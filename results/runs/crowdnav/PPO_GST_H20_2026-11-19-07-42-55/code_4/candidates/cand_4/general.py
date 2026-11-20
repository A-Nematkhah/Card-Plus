
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    components = {}
    # Distinct structure: removed action term (to fix inert constant signal), progress weight increased to 10.0 with max(0,delta) for stronger goal approach signal, success bonus +40, collision -15, comfort penalty milder at 0.8 threshold with -2.0 scaling (reduces negative corr).
    if self.pre_obs is not None:
        prev_goal_dist = np.linalg.norm(self.pre_obs[0:2] - self.pre_obs[4:6])
        progress = prev_goal_dist - self.robot.goal_distance
    else:
        progress = 0.0
    progress_r = 10.0 * np.maximum(0.0, progress)
    components['progress'] = progress_r
    if self.min_separation < 0.8:
        comfort_r = -2.0 * (0.8 - self.min_separation)
    else:
        comfort_r = 0.0
    components['comfort'] = comfort_r
    collision_r = -15.0 if self.min_separation < 0 else 0.0
    components['collision'] = collision_r
    success_r = 40.0 if self.robot.goal_distance <= self.robot.radius else 0.0
    components['success'] = success_r
    freeze_r = 0.0
    components['freeze'] = freeze_r
    total_reward = progress_r + comfort_r + collision_r + success_r + freeze_r
    return total_reward, components
