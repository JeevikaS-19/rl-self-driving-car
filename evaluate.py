import torch
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from framestack import FrameStackWrapper
from perception import CNNAttentionExtractor
from phase3 import SACActor

# Load target checkpoint (e.g. 99999 or any saved step)
checkpoint_path = "checkpoints/checkpoint_v7_final_1249999.pt"
print(f"Loading checkpoint from: {checkpoint_path}")
try:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
except FileNotFoundError:
    print("Checkpoint file not found. Running with randomly initialized networks for debugging.")
    checkpoint = None

cnn_extractor = CNNAttentionExtractor()
actor = SACActor(latent_dim=2058, action_dim=2)

if checkpoint is not None:
    cnn_extractor.load_state_dict(checkpoint["cnn_extractor"])
    actor.load_state_dict(checkpoint["actor"])

cnn_extractor.eval()
actor.eval()

# Configure environment and render settings
config = dict(
    use_render=True,
    num_scenarios=100, # vary the map/traffic layout across episodes
    start_seed=0, # base seed; MetaDrive will cycle through scenarios
    out_of_road_penalty=15.0,
    crash_vehicle_penalty=15.0,
    use_lateral_reward=True,
    vehicle_config=dict(lidar=dict(num_lasers=240, distance=50, num_others=4))
)
raw_env = MetaDriveEnv(config)
env = FrameStackWrapper(raw_env)

# Hyperparameters (Matching train-v2.py)
nav_scale = 0.1  # INTERVENTION B: Visual-Navigation Balance scaling factor

obs, info = env.reset()
step_count = 0

for i in range(5000):
    grid_tensor = torch.FloatTensor(obs["grid"]).unsqueeze(0)
    nav_tensor = torch.FloatTensor(obs["nav_info"]).unsqueeze(0)
    
    with torch.no_grad():
        latent = cnn_extractor(grid_tensor)
        
        # Apply Intervention B: Scale down nav_tensor
        scaled_nav_tensor = nav_tensor * nav_scale
        actor_input = torch.cat([latent, scaled_nav_tensor], dim=-1)
        
        # Set deterministic=True to bypass stochastic chattering and output smooth policy mean (mu)
        action, _ = actor(actor_input, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action.squeeze(0).numpy())
    raw_env.render()
    step_count += 1

    print(f"step={step_count} speed={raw_env.agent.speed:.2f} steering={info['steering']:.4f} action={action.squeeze(0).numpy()}")

    if terminated or truncated:
        print(f"Episode ended after {step_count} steps.")
        step_count = 0
        obs, info = env.reset()

raw_env.close()
