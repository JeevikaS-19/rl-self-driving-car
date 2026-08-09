from metadrive.envs import MetaDriveEnv
import numpy as np

env = MetaDriveEnv(dict(use_render=True, traffic_density=0.3))

obs, info = env.reset()

for step in range(500):
    action = [0.0, 0.5]  # steer straight, moderate throttle — adjust if needed
    obs, reward, terminated, truncated, info = env.step(action)

    lidar = obs[19:259]
    closest_idx = np.argmin(lidar)
    closest_val = lidar[closest_idx]

    # Only print when something is genuinely close, to avoid spam
    if closest_val < 0.9:
        print(f"Step {step} | closest ray index: {closest_idx} | distance value: {closest_val:.3f}")

    if terminated or truncated:
        obs, info = env.reset()

env.close()