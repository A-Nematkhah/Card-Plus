
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    components = {}
    # Distinct structure: clipped progress (max(0, delta)) with weight 4.0 to reward only goal reduction without penalizing flee steps (improves flee probe); collision scaled to -25 for strong separation (timeout ~0 > collision); success +25; comfort penalty-only below 0.5 to avoid farming; action small quadratic.
    if self.pre_obs is not None:
        prev_goal_dist = np.linalg.norm(self.pre_obs[0:2] - self.pre_obs[4:6])
        progress = prev_goal_dist - self.robot.goal_distance
    else:
        progress = 0.0
    progress_r = 4.0 * np.maximum(0.0, progress)
    components['progress'] = progress_r
    if self.min_separation < 0.5:
        comfort_r = -4.0 * (0.5 - self.min_separation)
    else:
        comfort_r = 0.0
    components['comfort'] = comfort_r
    collision_r = -25.0 if self.min_separation < 0 else 0.0
    components['collision'] = collision_r
    success_r = 25.0 if self.robot.goal_distance <= self.robot.radius else 0.0
    components['success'] = success_r
    action_r = -0.05 * np.sum(action ** 2)
    components['action'] = action_r
    freeze_r = 0.0
    components['freeze'] = freeze_r
    total_reward = progress_r + comfort_r + collision_r + success_r + action_r + freeze_r
    return total_reward, components
