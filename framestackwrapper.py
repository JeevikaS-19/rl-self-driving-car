import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
from framestack import FrameStackWrapper

env = MetaDriveEnv(dict(
    use_render=False,
    traffic_density=0.3,
    vehicle_config=dict(lidar=dict(num_others=4)),
))
wrapper = FrameStackWrapper(env)

obs_dict, info = wrapper.reset()
print("Actual resolved lidar config:", env.agent.config["lidar"])
print("num_others values at reset:", obs_dict["num_others"])

# drive a bit and check again, since reset() might just have no traffic nearby yet
for i in range(20):
    obs_dict, reward, terminated, truncated, info = wrapper.step([0.0, 1.0])

print("num_others values after driving:", obs_dict["num_others"])