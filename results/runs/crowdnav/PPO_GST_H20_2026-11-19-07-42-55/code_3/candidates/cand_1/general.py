
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    components = {}
    # Distinct structure with new weighting: progress uses max(0, delta) * 8.0 (stronger dense shaping for goal approach to boost success rate); comfort penalty scaled at 0.6 threshold with -5.0; collision -20 and success +20 for balanced separation; action -0.03 * norm^2.
    if self.pre_obs is not None:
        prev_goal_dist = np.linalg.norm(self.pre_obs[0:2] - self.pre_obs[4:6])
        progress = prev_goal_dist - self.robot.goal_distance
    else:
        progress = 0.0
    progress_r = 8.0 * np.maximum(0.0, progress)
    components['progress'] = progress_r
    if self.min_separation < 0.6:
        comfort_r = -5.0 * (0.6 - self.min_separation)
    else:
        comfort_r = 0.0
    components['comfort'] = comfort_r
    collision_r = -20.0 if self.min_separation < 0 else 0.0
    components['collision'] = collision_r
    success_r = 20.0 if self.robot.goal_distance <= self.robot.radius else 0.0
    components['success'] = success_r
    action_r = -0.03 * np.sum(action ** 2)
    components['action'] = action_r
    freeze_r = 0.0
    components['freeze'] = freeze_r
    total_reward = progress_r + comfort_r + collision_r + success_r + action_r + freeze_r
    return total_reward, components
