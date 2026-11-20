
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    components = {}
    # Distinct structure: progress only positive deltas via np.maximum scaled by 10.0 (strong goal progress encouragement without punishing flee to keep timeout return > collision); comfort penalty at 0.5m threshold with -5.0 scale; terminals rescaled to -10/+25 for better timeout/collided ratio and separation margin; action quadratic at -0.08.
    if self.pre_obs is not None:
        prev_goal_dist = np.linalg.norm(self.pre_obs[0:2] - self.pre_obs[4:6])
        progress = prev_goal_dist - self.robot.goal_distance
    else:
        progress = 0.0
    progress_r = 10.0 * np.maximum(0.0, progress)
    components['progress'] = progress_r
    if self.min_separation < 0.5:
        comfort_r = -5.0 * (0.5 - self.min_separation)
    else:
        comfort_r = 0.0
    components['comfort'] = comfort_r
    collision_r = -10.0 if self.min_separation < 0 else 0.0
    components['collision'] = collision_r
    success_r = 25.0 if self.robot.goal_distance <= self.robot.radius else 0.0
    components['success'] = success_r
    action_r = -0.08 * np.sum(action ** 2)
    components['action'] = action_r
    freeze_r = 0.0
    components['freeze'] = freeze_r
    total_reward = progress_r + comfort_r + collision_r + success_r + action_r + freeze_r
    return total_reward, components
