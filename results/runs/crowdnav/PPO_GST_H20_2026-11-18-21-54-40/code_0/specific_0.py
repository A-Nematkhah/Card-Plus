
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    prev_pos = self.pre_obs[0:2]
    goal = obs[4:6]
    prev_dist = np.linalg.norm(prev_pos - goal)
    delta = prev_dist - float(np.linalg.norm(obs[0:2] - obs[4:6]))
    progress = delta * 1.1
    if delta < 0.0:
        progress -= 0.9 * abs(delta)
    min_d = min(obs[7], obs[13], obs[14])
    if min_d < 0.0:
        clearance = -50.0
    elif min_d < 0.3:
        clearance = -5.5 * (0.3 - min_d)
    else:
        clearance = 0.0
    if float(np.linalg.norm(obs[2:4])) < 0.08 and float(np.linalg.norm(obs[0:2] - obs[4:6])) > 0.7:
        freeze = -0.45
    else:
        freeze = 0.0
    act = -0.015 * np.sum(action**2)
    reward = progress + clearance + freeze + act
    info = {"progress": progress, "clearance": clearance, "freeze": freeze, "act": act}
    return reward, info
