
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    is_collision = obs[7] <= 0.0
    is_success = float(np.linalg.norm(obs[0:2] - obs[4:6])) <= obs[6]
    components = {'progress': 0.0, 'comfort': 0.0, 'action_reg': 0.0, 'freeze': 0.0, 'collision': 0.0, 'success': 0.0}
    if is_collision:
        reward = -20.0
        components['collision'] = -20.0
        return reward, components
    if is_success:
        reward = 20.0
        components['success'] = 20.0
        return reward, components
    cur_pos = obs[0:2]
    cur_goal = obs[4:6]
    cur_d = ((cur_pos[0] - cur_goal[0])**2 + (cur_pos[1] - cur_goal[1])**2)**0.5
    prev_pos = self.pre_obs[0:2]
    prev_goal = self.pre_obs[4:6]
    prev_d = ((prev_pos[0] - prev_goal[0])**2 + (prev_pos[1] - prev_goal[1])**2)**0.5
    progress = prev_d - cur_d
    # Modified: progress weight set to 5.5 (reason: distinct weighting from base to strengthen dense progress signal while avoiding harmful reduction)
    progress_reward = 5.5 * progress
    components['progress'] = progress_reward
    # Modified: comfort threshold=0.9 with quadratic coeff=-1.8 using obs[7] (reason: distinct structure and slight rescaling to address [HIGHER_IN_FAILURES] comfort without repeating prior thresholds or additions)
    comfort_threshold = 0.9
    min_sep = obs[7]
    if min_sep < comfort_threshold:
        comfort_penalty = -1.8 * (comfort_threshold - min_sep)**2
    else:
        comfort_penalty = 0.0
    components['comfort'] = comfort_penalty
    # Modified: action_reg coeff to -0.09 (reason: distinct regularization scaling to mitigate [NEG_CORR] action_reg and [HIGHER_IN_FAILURES])
    action_reg = -0.09 * (action[0]**2 + action[1]**2)
    components['action_reg'] = action_reg
    # Modified: freeze uses self.robot attributes with speed<0.3, dist>radius+0.6, penalty=-0.18 (reason: distinct freeze parameters using listed float(np.linalg.norm(obs[2:4])) and float(np.linalg.norm(obs[0:2] - obs[4:6])) to improve anti-freeze without harmful prior changes)
    freeze_penalty = 0.0
    if float(np.linalg.norm(obs[0:2] - obs[4:6])) > obs[6] + 0.6 and float(np.linalg.norm(obs[2:4])) < 0.3:
        freeze_penalty = -0.18
    components['freeze'] = freeze_penalty
    reward = progress_reward + comfort_penalty + action_reg + freeze_penalty
    return reward, components
