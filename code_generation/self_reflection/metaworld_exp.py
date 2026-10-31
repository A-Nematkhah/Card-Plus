# Task instructions and observation mappings used by CARD Meta-World.
# This file is imported by metaworld_exp_one_step.py / query_llm_metaworld.py
# but was missing from the public CARD repo.

instruction_mapping = {
    "drawer-open-v2": "Open a drawer by its handle.",
    "drawer-close-v2": "Close the drawer.",
    "window-open-v2": "Open the window.",
    "window-close-v2": "Close the window.",
    "button-press-v2": "Press the button.",
    "sweep-into-v2": "Sweep the puck into the hole.",
    "door-unlock-v2": "Unlock the door by rotating the lock counterclockwise.",
    "door-close-v2": "Close the door.",
    "handle-press-v2": "Press the handle down.",
    "handle-press-side-v2": "Press the handle down from the side.",
}

# Abstract LLM names -> expressions that exist in compute_dense_reward / collect_trajectories.
# collect_trajectories eval()s these (except self.goal_position, which is read from info).
mapping_dicts = {
    "self.robot.ee_position": "obs[:3]",
    "self.robot.gripper_openness": "obs[3]",
    "self.obj1.position": "obs[4:7]",
    "self.obj1.quaternion": "obs[7:11]",
    "self.goal_position": "self.env._get_pos_goal()",
}
