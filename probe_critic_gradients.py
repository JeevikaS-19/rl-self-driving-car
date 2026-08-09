import torch
import numpy as np
from framestack import FrameStackWrapper
from metadrive.envs.metadrive_env import MetaDriveEnv
from phase3 import SACCritic
from perception import CNNAttentionExtractor

def probe_gradients():
    # 1. Initialize environment
    config = dict(
        use_render=False,
        out_of_road_penalty=15.0,
        crash_vehicle_penalty=15.0,
        use_lateral_reward=True,
        vehicle_config=dict(lidar=dict(num_lasers=240, distance=50, num_others=4))
    )
    raw_env = MetaDriveEnv(config)
    env = FrameStackWrapper(raw_env)

    # 2. Load checkpoint
    checkpoint_path = "checkpoints/checkpoint_v5_final_99999.pt"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    cnn_extractor = CNNAttentionExtractor()
    critic = SACCritic(latent_dim=2048, cheat_dim=16, action_dim=2)

    cnn_extractor.load_state_dict(checkpoint["cnn_extractor"])
    critic.load_state_dict(checkpoint["critic"])

    cnn_extractor.eval()
    critic.eval()

    # 3. Get typical starting observation
    obs, _ = env.reset()
    grid_tensor = torch.FloatTensor(obs["grid"]).unsqueeze(0)
    num_others_tensor = torch.FloatTensor(obs["num_others"]).unsqueeze(0)

    # 4. Extract representation
    with torch.no_grad():
        latent_vector = cnn_extractor(grid_tensor)
        obs_critic_base = torch.cat([latent_vector, num_others_tensor], dim=-1)

    print("\n" + "="*60)
    print("PROBING CRITIC SPATIAL GRADIENTS: dQ/da")
    print("="*60)

    # 5. Define an action tensor and enable gradient tracking on the action input!
    # Let's test a zero action (steering=0, throttle=0)
    action_array = np.array([[0.0, 0.0]], dtype=np.float32)
    action_tensor = torch.tensor(action_array, requires_grad=True)

    # 6. Forward pass through Twin Critic 1
    q1, q2 = critic(obs_critic_base, action_tensor)

    # 7. Backpropagate to extract the analytical gradient of Q1 with respect to action
    q1.backward()

    # 8. Extract the gradients
    dq_da = action_tensor.grad.squeeze(0).numpy()

    print(f"Evaluated State Q1-Value : {q1.item():.4f}")
    print(f"Analytical Gradient (dQ/da) : Steering: {dq_da[0]:.8f} | Throttle/Brake: {dq_da[1]:.8f}")
    print("="*60)

    env.close()

if __name__ == "__main__":
    probe_gradients()