
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    components = {}
    # New distinct structure: progress uses 6.0 * max(0, delta) to focus signal on positive movement only; collision rescaled to -5.0 (addresses dominant collision mass drowning progress); comfort penalty milder at -2.0 scale below 0.5m; terminals +/-20; action cost reduced to -0.05 * sum sq; this balances terms for better sample efficiency while keeping success/collision large but not overwhelming.
    if self.pre_obs is not None:
        prev_goal_dist = np.linalg.norm(self.pre_obs[0:2] - self.pre_obs[4:6])
        progress = prev_goal_dist - self.robot.goal_distance
    else:
        progress = 0.0
    progress_r = 6.0 * np.maximum(0.0, progress)
    components['progress'] = progress_r
    if self.min_separation < 0.5:
        comfort_r = -2.0 * (0.5 - self.min_separation)
    else:
        comfort_r = 0.0
    components['comfort'] = comfort_r
    # Re-scaled collision from -30 to -5 to reduce its 70% dominance and allow more exploration toward goal
    collision_r = -5.0 if self.min_separation < 0 else 0.0
    components['collision'] = collision_r
    success_r = 20.0 if self.robot.goal_distance <= self.robot.radius else 0.0
    components['success'] = success_r
    action_r = -0.05 * np.sum(action ** 2)
    components['action'] = action_r
    freeze_r = 0.0
    components['freeze'] = freeze_r
    total_reward = progress_r + comfort_r + collision_r + success_r + action_r + freeze_r
    return total_reward, components
