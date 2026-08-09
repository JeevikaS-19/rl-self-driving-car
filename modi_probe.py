import torch
import numpy as np
import pandas as pd
from framestack import FrameStackWrapper
from metadrive.envs.metadrive_env import MetaDriveEnv
from phase3 import SACCritic, SACActor
from perception import CNNAttentionExtractor
#for turning 
def probe_critic_at_turn(max_steps=2000, checkpoint_path="checkpoints/checkpoint_v6_final_99999.pt"):
    config = dict(
        use_render=False,
        out_of_road_penalty=15.0,
        crash_vehicle_penalty=15.0,
        use_lateral_reward=True,
        num_scenarios=200,
        vehicle_config=dict(lidar=dict(num_lasers=240, distance=50, num_others=4))
    )
    raw_env = MetaDriveEnv(config)
    env = FrameStackWrapper(raw_env)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    cnn_extractor = CNNAttentionExtractor()
    critic = SACCritic(latent_dim=2058, cheat_dim=16, action_dim=2)
    actor = SACActor(latent_dim=2058, action_dim=2)

    cnn_extractor.load_state_dict(checkpoint["cnn_extractor"])
    critic.load_state_dict(checkpoint["critic"])
    actor.load_state_dict(checkpoint["actor"])

    cnn_extractor.eval()
    critic.eval()
    actor.eval()

    obs, info = env.reset()
    found_turn = False

    for i in range(max_steps):
        grid_tensor = torch.FloatTensor(obs["grid"]).unsqueeze(0)
        nav_tensor = torch.FloatTensor(obs["nav_info"]).unsqueeze(0)
        with torch.no_grad():
            latent = cnn_extractor(grid_tensor)
            actor_input = torch.cat([latent, nav_tensor], dim=-1)
            action, _ = actor(actor_input, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action.squeeze(0).numpy())

        nav_command = info.get("navigation_command", "unknown")

        if nav_command != "forward":
            print(f"Found turn at step {i}: navigation_command = '{nav_command}'")
            found_turn = True
            break

        if terminated or truncated:
            print(f"Episode ended at step {i} (still 'forward', never reached a turn). Resetting.")
            obs, info = env.reset()

    if not found_turn:
        print(f"No turn encountered within {max_steps} steps. Try increasing max_steps or a different checkpoint.")
        env.close()
        return

    print(f"Probing Q-surface AT the turn (navigation_command='{nav_command}')")

    grid_tensor = torch.FloatTensor(obs["grid"]).unsqueeze(0)
    num_others_tensor = torch.FloatTensor(obs["num_others"]).unsqueeze(0)
    nav_info_tensor = torch.FloatTensor(obs["nav_info"]).unsqueeze(0)

    with torch.no_grad():
        latent_vector = cnn_extractor(grid_tensor)
        obs_critic_base = torch.cat([latent_vector, nav_info_tensor, num_others_tensor], dim=-1)

    steer_grid = np.linspace(-1.0, 1.0, 11)
    throttle_grid = np.linspace(-1.0, 1.0, 11)

    q_grid_1 = np.zeros((11, 11))
    for i, steer in enumerate(steer_grid):
        for j, throttle in enumerate(throttle_grid):
            action_t = torch.FloatTensor([[steer, throttle]])
            with torch.no_grad():
                q1, _ = critic(obs_critic_base, action_t)
                q_grid_1[i, j] = q1.item()

    df_q1 = pd.DataFrame(
        q_grid_1,
        index=[f"Steer={s:.2f}" for s in steer_grid],
        columns=[f"Throt={t:.2f}" for t in throttle_grid]
    )
    print("\n--- Q-VALUES AT TURN (ROWS=STEER, COLS=THROTTLE) ---")
    print(df_q1.round(4))

    best_idx = np.unravel_index(np.argmax(q_grid_1), q_grid_1.shape)
    print(f"\nBest action according to critic: Steer={steer_grid[best_idx[0]]:.2f}, Throttle={throttle_grid[best_idx[1]]:.2f}")
    print(f"Expected direction based on nav_command '{nav_command}': {'steer negative (left)' if nav_command=='left' else 'steer positive (right)' if nav_command=='right' else 'unclear'}")

    raw_env.close()

if __name__ == "__main__":
    try:
        probe_critic_at_turn()
    except Exception as e:
        print(f"Diagnostic Error: {e}")