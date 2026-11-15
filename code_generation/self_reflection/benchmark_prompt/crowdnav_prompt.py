from code_generation.self_reflection.crowdnav_env_utils import SENSOR_RANGE, describe_action_space

intro = f"""
You are an expert in robotics, reinforcement learning and code generation.
We are going to use a holonomic mobile robot to complete given navigation tasks in the CrowdNav++ / EvoNav circle-crossing simulator (~12×12 m, 5 ORCA pedestrians that react to the robot; robot is invisible to humans). {describe_action_space()}

Now I want you to help me write a reward function for reinforcement learning.
Typically, the reward function of a crowd-navigation task consists of these parts (include a part only if it is necessary):
1. progress toward the goal (distance reduction vs previous step)
2. collision / near-collision penalties with pedestrians
3. comfort / clearance terms (penalty only when too close — never a bonus for being far)
4. regularization of the robot's action or speed
...
""".strip()

environment_description = f"""
class BaseEnv(gym.Env):
    self.robot : Robot
    self.pre_obs : np.ndarray  # flat observation from the PREVIOUS step (same layout as obs)
    # Humans are dynamic agents controlled by ORCA. Prefer the nearest-human features below;
    # do not invent attributes that are not listed.

class Robot:
    self.position : np.ndarray[(2,)]  # robot (px, py) in meters
    self.velocity : np.ndarray[(2,)]  # robot (vx, vy) in m/s
    self.goal_position : np.ndarray[(2,)]  # goal (gx, gy)
    self.radius : float
    self.goal_distance : float  # Euclidean distance from robot to goal
    self.speed : float  # ||velocity||

# Scalar / vector features available in the reward function (already computed from the observation):
self.min_separation : float  # clearance to the closest human (center distance minus radii)
self.nearest_human.distance : float
self.nearest_human.relative_position : np.ndarray[(2,)]  # human - robot
self.nearest_human.relative_velocity : np.ndarray[(2,)]
self.second_nearest_human.distance : float
self.third_nearest_human.distance : float
self.detected_human_num : float

# Flat obs layout (use these indices when reading obs / self.pre_obs):
# obs[0:2] robot pos, obs[2:4] vel, obs[4:6] goal, obs[6] radius, obs[7] min_separation

Sensor range is about {SENSOR_RANGE} meters. Prefer distances and relative quantities that are listed above.
You are allowed to use any existing python package if applicable, but only when necessary.
""".strip()

additional_knowledge = """
Additional knowledge (HARD RULES — violating them fails the offline gate):
1. Success means reaching the goal without colliding with any pedestrian.
2. Use a LARGE positive terminal reward for success and a LARGE negative terminal penalty for collision.
   Dense per-step terms must be much smaller in magnitude than these terminals (|dense| << |terminal|).
3. Progress MUST be delta distance to goal vs the previous step, using self.pre_obs, e.g.
   cur_d = ||obs[0:2] - obs[4:6]||; prev_d = ||self.pre_obs[0:2] - self.pre_obs[4:6]||; progress = prev_d - cur_d
   (positive when moving closer). Do NOT use only -k * current_distance as the main progress signal.
4. Clearance / min_separation: PENALTY ONLY when too close (e.g. if obs[7] < comfort: negative term).
   NEVER add a positive dense bonus proportional to obs[7] / min_separation (that causes fleeing / clearance farming).
5. NEVER reward the robot for moving farther from humans or farther from the goal.
6. Do not invent sensors, maps, or human IDs that are not listed.
""".strip()

zero_shot_example = ""

zero_shot_hint = """
I want it to fulfill the following task: {instruction}
1. Please think step by step and tell me what does this task mean;
2. Then write a function that strictly follows the following format, and the return value of the function strictly follows the Type Annotation.
`
def compute_dense_reward(self, action: np.ndarray, obs: np.ndarray) -> Tuple[float, Dict[str, float]]:
    ...
    return reward, {{...}}
`
The output of the reward function should consist of two items:
    (1) the total reward,
    (2) a dictionary of each individual reward component.
The code output should be formatted as a python code string: "```python ... ```". Just the function body is fine.
3. Do not invent any variable or attribute that is not given. Always use the listed names exactly (e.g. self.robot.goal_distance, never self.goal_distance).
4. Required: delta progress via self.pre_obs; clearance penalty-only; large success/collision terminals; no positive bonus for larger separation.
5. When you writing code, you can also add some comments as your thought.
""".strip()

crowdnav_system_prompt = intro + "\n\n" + environment_description + "\n\n" + additional_knowledge
crowdnav_first_generation_prompt = zero_shot_hint
