
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    components = {}
    # New structure: clipped progress (only positive deltas, weight 8.0) to reward goal approach without punishing flee (fixes negative flee probe); collision large negative -30 for separation (timeout ~0 >> collision); success +30; comfort penalty-only below 0.5 with -5.0 (no farming); action small quadratic.
    if self.pre_obs is not None:
        prev_goal_dist = np.linalg.norm(self.pre_obs[0:2] - self.pre_obs[4:6])
        progress = prev_goal_dist - self.robot.goal_distance
    else:
        progress = 0.0
    progress_r = 8.0 * np.maximum(0.0, progress)
    components['progress'] = progress_r
    if self.min_separation < 0.5:
        comfort_r = -5.0 * (0.5 - self.min_separation)
    else:
        comfort_r = 0.0
    components['comfort'] = comfort_r
    collision_r = -30.0 if self.min_separation < 0 else 0.0
    components['collision'] = collision_r
    success_r = 30.0 if self.robot.goal_distance <= self.robot.radius else 0.0
    components['success'] = success_r
    action_r = -0.05 * np.sum(action ** 2)
    components['action'] = action_r
    total_reward = progress_r + comfort_r + collision_r + success_r + action_r
    return total_reward, components
