import torch  # always first, avoids the c10.dll conflict
from metadrive.envs.metadrive_env import MetaDriveEnv

env = MetaDriveEnv(dict(use_render=False))
print("Lateral Reward Enabled:", env.config["use_lateral_reward"])
env.close()