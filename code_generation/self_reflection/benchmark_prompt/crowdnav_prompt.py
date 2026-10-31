from code_generation.self_reflection.crowdnav_env_utils import SENSOR_RANGE, describe_action_space

intro = f"""
You are an expert in robotics, reinforcement learning and code generation.
We are going to use a holonomic mobile robot to complete given navigation tasks in the CrowdNav++ / EvoNav circle-crossing simulator (~12×12 m, 20 ORCA pedestrians that react to the robot). {describe_action_space()}

Now I want you to help me write a reward function for reinforcement learning.
Typically, the reward function of a crowd-navigation task consists of these parts (include a part only if it is necessary):
1. progress toward the goal (distance reduction)
2. collision / near-collision penalties with pedestrians
3. comfort / clearance terms (minimum separation)
4. regularization of the robot's action or speed
...
""".strip()

environment_description = f"""
class BaseEnv(gym.Env):
    self.robot : Robot
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

Sensor range is about {SENSOR_RANGE} meters. Prefer distances and relative quantities that are listed above.
You are allowed to use any existing python package if applicable, but only when necessary.
""".strip()

additional_knowledge = """
Additional knowledge:
1. Success means reaching the goal without colliding with any pedestrian.
2. A large positive terminal reward for success and a large negative terminal penalty for collision are common.
3. Dense shaping should encourage moving closer to the goal while keeping a safe clearance.
4. Do not invent sensors, maps, or human IDs that are not listed.
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
3. Do not invent any variable or attribute that is not given.
4. When you writing code, you can also add some comments as your thought.
""".strip()

crowdnav_system_prompt = intro + "\n\n" + environment_description + "\n\n" + additional_knowledge
crowdnav_first_generation_prompt = zero_shot_hint
