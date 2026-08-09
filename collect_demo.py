import torch
from metadrive.envs.metadrive_env import MetaDriveEnv
from framestack import FrameStackWrapper
import pickle
import numpy as np

config = dict(
    use_render=False,
    manual_control=True,
    out_of_road_penalty=15.0,
    crash_vehicle_penalty=15.0,
    use_lateral_reward=True,
    num_scenarios=200,
    vehicle_config=dict(lidar=dict(num_lasers=240, distance=50, num_others=4))
)
raw_env = MetaDriveEnv(config)
env = FrameStackWrapper(raw_env)

demonstrations = []
num_episodes = 100

for ep in range(num_episodes):
    obs, info = env.reset()
    raw_env.agent.expert_takeover = True

    while True:
        action = [0.0, 0.0]  # placeholder — expert_takeover overrides actual control internally
        next_obs, reward, terminated, truncated, info = env.step(action)

        real_action = np.array(info.get("raw_action", action), dtype=np.float32)

        demonstrations.append({
            "obs_grid": obs["grid"].numpy(),
            "obs_num_others": obs["num_others"].numpy(),
            "obs_nav_info": obs["nav_info"].numpy(),
            "action": real_action,
            "reward": reward,
            "next_obs_grid": next_obs["grid"].numpy(),
            "next_obs_num_others": next_obs["num_others"].numpy(),
            "next_obs_nav_info": next_obs["nav_info"].numpy(),
            "terminated": terminated,
            "truncated": truncated,
        })

        obs = next_obs

        if terminated or truncated:
            break

    print(f"Episode {ep+1}/{num_episodes} collected, total transitions so far: {len(demonstrations)}")

with open("demonstrations.pkl", "wb") as f:
    pickle.dump(demonstrations, f)

print(f"Saved {len(demonstrations)} demonstration transitions to demonstrations.pkl")
env.close()