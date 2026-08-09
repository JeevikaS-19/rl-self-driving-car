import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
from phase1 import projection_lidar_to_bev

env = MetaDriveEnv(dict(use_render=False, traffic_density=0.3))
obs, info = env.reset()

for i in range(20):
    obs, reward, terminated, truncated, info = env.step([0.0, 1.0])

lidar = obs[19:259]
print("min lidar value:", lidar.min())

grid = projection_lidar_to_bev(lidar)
print("grid sum (should be > 0 if something detected):", grid.sum())
print("grid dtype:", grid.dtype, "grid shape:", grid.shape)