import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
from framestack import FrameStackWrapper
import numpy as np

config = dict(
    use_render=True,
    manual_control=True,
    out_of_road_penalty=15.0,
    crash_vehicle_penalty=15.0,
    use_lateral_reward=True,
    num_scenarios=200,
    vehicle_config=dict(lidar=dict(num_lasers=240, distance=50, num_others=4))
)
raw_env = MetaDriveEnv(config)
env = FrameStackWrapper(raw_env)

obs, info = env.reset()
raw_env.agent.expert_takeover = True

for step in range(500):
    action = [0.0, 0.0]  # placeholder — expert_takeover should override this
    next_obs, reward, terminated, truncated, info = env.step(action)
    raw_env.render()

    real_action = np.array(info.get("raw_action", action), dtype=np.float32)
    print(f"step={step} speed={raw_env.agent.speed:.2f} placeholder=[0.0, 0.0] info_action={real_action}")
    print(info)
    obs = next_obs
    if terminated or truncated:
        obs, info = env.reset()
        raw_env.agent.expert_takeover = True

env.close()