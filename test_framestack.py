import torch  # keep first, avoids the DLL conflict
from metadrive.envs.metadrive_env import MetaDriveEnv
from framestack import FrameStackWrapper

env = MetaDriveEnv(dict(use_render=False, traffic_density=0.3))  # some traffic so there's something to detect
wrapper = FrameStackWrapper(env)

# --- Test reset ---
stacked, info = wrapper.reset()
print("Shape after reset:", stacked.shape)
print("Sum of grid at reset (0 = confirmed empty scene):", torch.sum(stacked).item())

all_same = (stacked[0] == stacked[1]).all() and (stacked[1] == stacked[2]).all() and (stacked[2] == stacked[3]).all()
print("All 4 frames identical after reset (expected True):", all_same.item())

# --- Drive forward with full throttle for several steps ---
for i in range(25):
    stacked2, reward, terminated, truncated, info = wrapper.step([0.0, 1.0])  # steering=0, throttle=max
    if terminated or truncated:
        break

print("Shape after driving:", stacked2.shape)
print("Sum of grid after driving:", torch.sum(stacked2).item())

newest_changed = not (stacked2[3] == stacked[3]).all()
oldest_still_matches_original_reset = (stacked2[0] == stacked[0]).all()
print("Newest frame changed after driving (expected True):", newest_changed)
print("Oldest frame from wrapper now different from original reset frame (expected True, since 20 steps have passed):", not oldest_still_matches_original_reset.item())

env.close()