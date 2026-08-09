from metadrive.envs.metadrive_env import MetaDriveEnv

env = MetaDriveEnv(dict(use_render=False))
obs, info = env.reset()
print(env.agent.config["lidar"])
print(type(obs), obs.shape if hasattr(obs, "shape") else len(obs))
print(obs[:20])
env.close()