
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    components = {}
    if self.pre_obs is not None:
        prev_goal_dist = np.linalg.norm(self.pre_obs[0:2] - self.pre_obs[4:6])
        progress = prev_goal_dist - self.robot.goal_distance
    else:
        progress = 0.0
    progress_r = 5.0 * np.maximum(0.0, progress)
    components['progress'] = progress_r
    if self.min_separation < 0.4:
        comfort_r = -6.0 * (0.4 - self.min_separation)
    else:
        comfort_r = 0.0
    components['comfort'] = comfort_r
    # Modified collision penalty from -30.0 to -8.0: collision was dominant (70% reward mass, drowning progress); smaller scale reduces over-penalization, lets dense progress guide toward goal while still discouraging collisions, improving sample efficiency and success rate.
    collision_r = -8.0 if self.min_separation < 0 else 0.0
    components['collision'] = collision_r
    success_r = 30.0 if self.robot.goal_distance <= self.robot.radius else 0.0
    components['success'] = success_r
    action_r = -0.1 * np.sum(action ** 2)
    components['action'] = action_r
    freeze_r = 0.0
    components['freeze'] = freeze_r
    total_reward = progress_r + comfort_r + collision_r + success_r + action_r + freeze_r
    return total_reward, components
