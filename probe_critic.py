import torch
import numpy as np
import pandas as pd
from framestack import FrameStackWrapper
from metadrive.envs.metadrive_env import MetaDriveEnv
from phase3 import SACCritic, SACActor
from perception import CNNAttentionExtractor

def probe_critic_surface():
    # 1. Initialize the exact environment with your reward shaping parameters
    config = dict(
        use_render=False,
        out_of_road_penalty=15.0,
        crash_vehicle_penalty=15.0,
        use_lateral_reward=True,
        vehicle_config=dict(lidar=dict(num_lasers=240, distance=50, num_others=4))
    )
    raw_env = MetaDriveEnv(config)
    env = FrameStackWrapper(raw_env)

    # 2. Load networks and checkpoints from your 50,000-step run
    checkpoint_path = "checkpoints/checkpoint_final_99999.pt"
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    cnn_extractor = CNNAttentionExtractor()
    critic = SACCritic(latent_dim=2048, cheat_dim=16, action_dim=2)
    actor = SACActor(latent_dim=2048, action_dim=2)

    cnn_extractor.load_state_dict(checkpoint["cnn_extractor"])
    critic.load_state_dict(checkpoint["critic"])
    actor.load_state_dict(checkpoint["actor"])

    cnn_extractor.eval()
    critic.eval()
    actor.eval()

    # 3. Extract a sample starting observation from the environment
    obs, info = env.reset()
    grid_tensor = torch.FloatTensor(obs["grid"]).unsqueeze(0)
    num_others_tensor = torch.FloatTensor(obs["num_others"]).unsqueeze(0)

    # 4. Compute the spatial representation (freeze gradients for inference)
    with torch.no_grad():
        latent_vector = cnn_extractor(grid_tensor)
        # Assemble the asymmetric critic input (2048 cnn latent + 16 cheat dims)
        obs_critic_base = torch.cat([latent_vector, num_others_tensor], dim=-1)

    print("\n" + "="*60)
    print("PROBING CRITIC Q-VALUE LANDSCAPE")
    print("="*60)

    # 5. Define action ranges to sample
    # Actions are normalized to [-1, 1] range: [Steering, Throttle/Brake]
    steer_grid = np.linspace(-1.0, 1.0, 11)   # Sweep 11 points from Hard Left to Hard Right
    throttle_grid = np.linspace(-1.0, 1.0, 11) # Sweep 11 points from Full Brake to Full Throttle

    q_grid_1 = np.zeros((11, 11))
    q_grid_2 = np.zeros((11, 11))

    # Evaluate the full 2D grid across twin critics
    for i, steer in enumerate(steer_grid):
        for j, throttle in enumerate(throttle_grid):
            action = torch.FloatTensor([[steer, throttle]])
            with torch.no_grad():
                q1, q2 = critic(obs_critic_base, action)
                q_grid_1[i, j] = q1.item()
                q_grid_2[i, j] = q2.item()

    # Format into a clean, readable DataFrame
    df_q1 = pd.DataFrame(
        q_grid_1, 
        index=[f"Steer={s:.2f}" for s in steer_grid], 
        columns=[f"Throt={t:.2f}" for t in throttle_grid]
    )
    
    print("\n--- TWIN CRITIC 1 Q-VALUES (ROWS=STEER, COLS=THROTTLE) ---")
    print(df_q1.round(4))

    # Calculate and output statistical telemetry
    flat_q1 = q_grid_1.flatten()
    flat_q2 = q_grid_2.flatten()

    print("\n" + "="*40)
    print("CRITIC STATISTICAL TELEMETRY")
    print("="*40)
    print(f"Critic 1 Q-value Range : [{flat_q1.min():.4f}, {flat_q1.max():.4f}]")
    print(f"Critic 1 Max-Min Delta : {flat_q1.max() - flat_q1.min():.4f}")
    print(f"Critic 1 Mean Q-value   : {flat_q1.mean():.4f}")
    print(f"Critic 1 Std Dev (σ_q)  : {flat_q1.std():.4f}")
    print("-" * 40)
    print(f"Critic 2 Q-value Range : [{flat_q2.min():.4f}, {flat_q2.max():.4f}]")
    print(f"Critic 2 Max-Min Delta : {flat_q2.max() - flat_q2.min():.4f}")
    print(f"Critic 2 Mean Q-value   : {flat_q2.mean():.4f}")
    print(f"Critic 2 Std Dev (σ_q)  : {flat_q2.std():.4f}")
    print("="*40)

    # 6. Specific extreme behaviors probing
    specific_actions = {
        "Default Frozen Action (Micro-Brake/Left)": [-0.023, -0.014],
        "Straight Standstill (Braking)          ": [0.000, -1.000],
        "Straight Cruising (Mild Throttle)      ": [0.000, 0.200],
        "Straight Maximum (Full Throttle)       ": [0.000, 1.000],
        "Hard Left (Max Throttle)               ": [-1.000, 1.000],
        "Hard Right (Max Throttle)              ": [1.000, 1.000],
    }

    print("\n--- SPECIFIC ACTION COMPARISON ---")
    for desc, act in specific_actions.items():
        act_tensor = torch.FloatTensor([act])
        with torch.no_grad():
            q1, q2 = critic(obs_critic_base, act_tensor)
            print(f"{desc} -> Q1: {q1.item():.4f} | Q2: {q2.item():.4f}")
    print("="*60)

    env.close()

if __name__ == "__main__":
    try:
        probe_critic_surface()
    except Exception as e:
        print(f"Diagnostic Error: {e}")
        print("Please ensure this script is run within your local 'project' folder where 'checkpoints/' is available.")