import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
from phase1 import projection_lidar_to_bev

env = MetaDriveEnv(dict(use_render=False, traffic_density=0.3))
obs, info = env.reset()

for i in range(20):
    obs, reward, terminated, truncated, info = env.step([0.0, 1.0])
    lidar = obs[19:259]
    grid = projection_lidar_to_bev(lidar)
    print(f"step={i} min_lidar={lidar.min():.3f} grid_sum={grid.sum():.1f} ego_speed={env.agent.speed:.2f}")
    if terminated or truncated:
        break

env.close()