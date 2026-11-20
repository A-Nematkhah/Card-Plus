
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    components = {}
    # Distinct structure: progress term uses only positive deltas via np.maximum(0, progress) with weight 5.0 (rewards getting closer, gives 0 on fleeing to avoid rewarding fleeing while encouraging goal progress); comfort penalty-only at threshold 0.4 with -6.0 scaling (no clearance farming); terminals at +/-30 for outcome separation; action cost small and constant.
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
    collision_r = -30.0 if self.min_separation < 0 else 0.0
    components['collision'] = collision_r
    success_r = 30.0 if self.robot.goal_distance <= self.robot.radius else 0.0
    components['success'] = success_r
    action_r = -0.1 * np.sum(action ** 2)
    components['action'] = action_r
    freeze_r = 0.0
    components['freeze'] = freeze_r
    total_reward = progress_r + comfort_r + collision_r + success_r + action_r + freeze_r
    return total_reward, components
