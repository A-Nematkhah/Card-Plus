import metaworld.envs.mujoco.env_dict as env_dict

env_cls = env_dict.ALL_V2_ENVIRONMENTS["drawer-open-v2"]
env = env_cls()
env._freeze_rand_vec = False
env._set_task_called = True
obs = env.reset()
action = env.action_space.sample()
obs2, reward, done, info = env.step(action)
print("drawer-open-v2 OK")
print("obs_shape:", getattr(obs, "shape", type(obs)))
print("reward:", float(reward))
print("success_key:", "success" in info)
env.close()
