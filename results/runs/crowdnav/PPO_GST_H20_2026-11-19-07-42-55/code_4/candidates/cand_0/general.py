
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    components = {}
    # Distinct structure with new weighting: progress max(0, delta) * 12.0 (stronger dense goal signal to drive success); comfort penalty milder at 0.7 threshold with -3.0 scaling (reduces negative corr with progress); collision -15, success +15, action -0.02 * norm^2.
    if self.pre_obs is not None:
        prev_goal_dist = np.linalg.norm(self.pre_obs[0:2] - self.pre_obs[4:6])
        progress = prev_goal_dist - self.robot.goal_distance
    else:
        progress = 0.0
    progress_r = 12.0 * np.maximum(0.0, progress)
    components['progress'] = progress_r
    if self.min_separation < 0.7:
        comfort_r = -3.0 * (0.7 - self.min_separation)
    else:
        comfort_r = 0.0
    components['comfort'] = comfort_r
    collision_r = -15.0 if self.min_separation < 0 else 0.0
    components['collision'] = collision_r
    success_r = 15.0 if self.robot.goal_distance <= self.robot.radius else 0.0
    components['success'] = success_r
    action_r = -0.02 * np.sum(action ** 2)
    components['action'] = action_r
    freeze_r = 0.0
    components['freeze'] = freeze_r
    total_reward = progress_r + comfort_r + collision_r + success_r + action_r + freeze_r
    return total_reward, components
