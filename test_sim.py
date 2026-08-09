import torch
from metadrive.envs.metadrive_env import MetaDriveEnv

env = MetaDriveEnv(dict(use_render=True, traffic_density=0.3))
obs, info = env.reset()

for i in range(100):
    obs, reward, terminated, truncated, info = env.step([0.0, 1.0])  # hardcoded, full throttle, no policy involved
    env.render()
    print(f"step={i} speed={env.agent.speed:.2f}")

env.close()