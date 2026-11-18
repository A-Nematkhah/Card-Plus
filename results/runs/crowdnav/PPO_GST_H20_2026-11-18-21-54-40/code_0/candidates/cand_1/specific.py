
import numpy as np
from typing import Tuple, Dict

def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    current_dist = float(np.linalg.norm(obs[0:2] - obs[4:6]))
    prev_dist = current_dist
    if self.pre_obs is not None:
        prev_dist = np.linalg.norm(self.pre_obs[0:2] - self.pre_obs[4:6])
    delta = prev_dist - current_dist
    
    components: Dict[str, float] = {}
    
    if current_dist <= obs[6] and obs[7] >= 0.0:
        total_reward = 100.0
        components['success'] = 100.0
    elif obs[7] < 0.0:
        total_reward = -100.0
        components['collision'] = -100.0
    else:
        progress_r = 4.0 * delta
        components['progress'] = progress_r
        sep = obs[7]
        comfort_r = -3.0 if sep < 0.3 else 0.0
        components['comfort'] = comfort_r
        action_r = -0.05 * np.sum(np.abs(action))
        components['action'] = action_r
        total_reward = progress_r + comfort_r + action_r
    
    return total_reward, components
