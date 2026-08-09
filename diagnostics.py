import torch
from framestack import FrameStackWrapper
from metadrive.envs.metadrive_env import MetaDriveEnv
from phase3 import SACActor
from perception import CNNAttentionExtractor

config = dict(
    use_render=True,
    vehicle_config=dict(lidar=dict(num_lasers=240, distance=50, num_others=4))
)
env = FrameStackWrapper(MetaDriveEnv(config))

checkpoint = torch.load("checkpoints/checkpoint_final_99999.pt")
cnn_extractor = CNNAttentionExtractor()
actor = SACActor(latent_dim=2048, action_dim=2)
cnn_extractor.load_state_dict(checkpoint["cnn_extractor"])
actor.load_state_dict(checkpoint["actor"])
cnn_extractor.eval()
actor.eval()

obs, _ = env.reset()
print("COMMENCING STANDARD DEVIATION DIAGNOSTIC")

for step in range(100):
    grid_tensor = torch.FloatTensor(obs["grid"]).unsqueeze(0)
    with torch.no_grad():
        latent_vector = cnn_extractor(grid_tensor)
        x = torch.relu(actor.linear1(latent_vector))
        x = torch.relu(actor.linear2(x))
        mu = actor.mu_head(x)
        log_std = torch.clamp(actor.log_std_head(x), actor.log_std_min, actor.log_std_max)
        std = torch.exp(log_std)
        action_tensor, _ = actor(latent_vector, deterministic=True)
        action = action_tensor.squeeze(0).cpu().numpy()

    next_obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step:03d} | action={action} | mu={mu.squeeze(0).numpy()} | std={std.squeeze(0).numpy()}")
    obs = next_obs
    if terminated or truncated:
        break

env.close()