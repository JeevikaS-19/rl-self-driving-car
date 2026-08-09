import torch
import torch.nn.functional as F
import numpy as np
import copy
from metadrive.envs.metadrive_env import MetaDriveEnv
from framestack import FrameStackWrapper
from perception import CNNAttentionExtractor
from phase3 import SACActor, SACCritic
from phase4 import ReplayBuffer
import os

# Create checkpoints directory
os.makedirs("checkpoints", exist_ok=True)

# =====================================================================
# 1. Initialize the Environment with the 16-dim num_others activated
# =====================================================================
config = dict(
    use_render=False,
    out_of_road_penalty=15.0,
    crash_vehicle_penalty=15.0,
    use_lateral_reward=True,
    vehicle_config=dict(lidar=dict(num_lasers=240, distance=50, num_others=4))
)
raw_env = MetaDriveEnv(config)
env = FrameStackWrapper(raw_env)

# =====================================================================
# 2. Instantiate the Memory Buffer
# =====================================================================
replay_buffer = ReplayBuffer(capacity=200000)

# =====================================================================
# 3. Instantiate the Neural Network Modules
# =====================================================================
cnn_extractor = CNNAttentionExtractor()
actor = SACActor(latent_dim=2058, action_dim=2)
critic = SACCritic(latent_dim=2058, cheat_dim=16, action_dim=2)

# =====================================================================
# 4. Instantiate Target Networks for Polyak Averaging
# =====================================================================
target_cnn_extractor = copy.deepcopy(cnn_extractor)
target_critic = copy.deepcopy(critic)

# Learnable entropy temperature setup
target_entropy = -2.0
log_alpha = torch.tensor(np.log(0.3), requires_grad=True, dtype=torch.float32)

# =====================================================================
# 5. Setup Optimizers
# =====================================================================
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
cnn_optimizer = torch.optim.Adam(cnn_extractor.parameters(), lr=3e-4)
alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4)

# =====================================================================
# 6. Hyperparameters (MODIFIED FOR INTERVENTIONS B & C)
# =====================================================================
max_steps = 100000         # INTERVENTION C: Extended Training Volume (100k steps)
warmup_steps = 5000        # Random exploration steps to populate replay buffer
update_frequency = 4       # Interval of global steps between optimization updates
batch_size = 256           # Sample size for stochastic gradient descent
gamma = 0.90               # Compressed horizon to prevent baseline state-value dominance
tau = 0.005                # Polyak target update coefficient
nav_scale = 0.3            # INTERVENTION B: Visual-Navigation Balance scaling factor

# =====================================================================
# 7. The Optimization Engine
# =====================================================================
def update_networks(batch_size, global_step):
    # Sample a mini-batch from Replay Buffer
    (batch_grid, batch_num_others, batch_nav_info, batch_actions, batch_rewards,
     batch_next_grid, batch_next_num_others, batch_next_nav_info, batch_dones) = replay_buffer.sample(batch_size)
    
    alpha = log_alpha.exp()

    # -----------------------------------------------------------------
    # A. Compute the SAC Bellman Target (No Gradients)
    # -----------------------------------------------------------------
    with torch.no_grad():
        next_latent = target_cnn_extractor(batch_next_grid)
        
        # Apply Intervention B: Scale down next nav info
        scaled_batch_next_nav_info = batch_next_nav_info * nav_scale
        next_obs_actor = torch.cat([next_latent, scaled_batch_next_nav_info], dim=-1)
        next_action, next_log_prob = actor(next_obs_actor, deterministic=False)
        
        next_obs_critic = torch.cat([next_latent, scaled_batch_next_nav_info, batch_next_num_others], dim=-1)
        target_q1, target_q2 = target_critic(next_obs_critic, next_action)
        min_target_q = torch.min(target_q1, target_q2)
        
        # Bellman Target Equation
        target_q = batch_rewards + gamma * (1.0 - batch_dones) * (min_target_q - alpha * next_log_prob)

    # -----------------------------------------------------------------
    # B. Update the Critic and CNN Extractor
    # -----------------------------------------------------------------
    latent = cnn_extractor(batch_grid)
    
    # Apply Intervention B: Scale down current nav info
    scaled_batch_nav_info = batch_nav_info * nav_scale
    obs_critic = torch.cat([latent, scaled_batch_nav_info, batch_num_others], dim=-1)
    
    current_q1, current_q2 = critic(obs_critic, batch_actions)
    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
    
    critic_optimizer.zero_grad()
    cnn_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    cnn_optimizer.step()

    # -----------------------------------------------------------------
    # C. Update the Actor (Policy Search)
    # -----------------------------------------------------------------
    # Detach visual features to prevent policy gradients from altering perception kernels
    detached_latent = latent.detach()
    scaled_batch_nav_info_detached = scaled_batch_nav_info.detach()
    
    # Apply Intervention B: Scale down nav info in policy evaluation
    fresh_obs_actor = torch.cat([detached_latent, scaled_batch_nav_info_detached], dim=-1)
    fresh_action, fresh_log_prob = actor(fresh_obs_actor, deterministic=False)
    
    fresh_obs_critic = torch.cat([detached_latent, scaled_batch_nav_info_detached, batch_num_others], dim=-1)
    actor_q1, actor_q2 = critic(fresh_obs_critic, fresh_action)
    actor_q = torch.min(actor_q1, actor_q2)
    
    actor_loss = (alpha * fresh_log_prob - actor_q).mean()
    
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # -----------------------------------------------------------------
    # D. Update the Dynamic Entropy Temperature (Alpha)
    # -----------------------------------------------------------------
    # Detach log-probability to isolate dual temperature gradients from policy heads
    alpha_loss = -(log_alpha.exp() * (fresh_log_prob.detach() + target_entropy)).mean()

    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    # -----------------------------------------------------------------
    # E. Target Network Polyak Averaging (Soft Updates)
    # -----------------------------------------------------------------
    with torch.no_grad():
        for target_param, live_param in zip(target_critic.parameters(), critic.parameters()):
            target_param.data.copy_(tau * live_param.data + (1.0 - tau) * target_param.data)
            
        for target_param, live_param in zip(target_cnn_extractor.parameters(), cnn_extractor.parameters()):
            target_param.data.copy_(tau * live_param.data + (1.0 - tau) * target_param.data)

# =====================================================================
# 8. Training Loop with Step B (Comfort & Smoothness Penalty)
# =====================================================================
obs, info = env.reset()
step_count = 0

# Initialize previous action tracker to zeros to begin smoothness penalty tracking
prev_action = np.zeros(2, dtype=np.float32)

print("\n" + "="*60)
print("LAUNCHING TRAINING PIPELINE (INTERVENTIONS B & C ACTIVE)")
print("="*60)

for global_step in range(max_steps):
    # A. Action Selection: Warmup Phase vs. Active Stochastic Policy
    if global_step < warmup_steps:
        # Uniform sampling to populate experience replay with high-entropy transitions
        # We query raw_env directly to bypass the FrameStackWrapper delegation bottleneck!
        action = raw_env.action_space.sample()
    else:
        grid_tensor = torch.FloatTensor(obs["grid"]).unsqueeze(0)
        nav_tensor = torch.FloatTensor(obs["nav_info"]).unsqueeze(0)
        with torch.no_grad():
            latent = cnn_extractor(grid_tensor)
            
            # Apply Intervention B: Scale down nav_tensor
            scaled_nav_tensor = nav_tensor * nav_scale
            actor_input = torch.cat([latent, scaled_nav_tensor], dim=-1)
            
            # Sample stochastically during training to maintain exploration pressure
            action_tensor, _ = actor(actor_input, deterministic=False)
            action = action_tensor.squeeze(0).numpy()

    # B. Execute Action in MetaDrive Environment
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # C. STEP B: Calculate Action-Smoothness (Comfort) Penalties
    # Penalize rapid, high-frequency counter-steering and violent throttle/brake jumps
    steer_change = (action[0] - prev_action[0]) ** 2
    throt_change = (action[1] - prev_action[1]) ** 2
    
    # Scale comfort penalties (standard control multiplier w_comfort = -0.1)
    comfort_penalty = -0.1 * steer_change - 0.1 * throt_change
    modified_reward = reward + comfort_penalty

    # D. Store Grounded, Modified Transition in Replay Buffer
    # We pass terminated and truncated separately to satisfy the ReplayBuffer signature
    # and we store the raw nav_info (scaling is applied dynamically during updates)
    replay_buffer.add(
        obs["grid"],
        obs["num_others"],
        obs["nav_info"],
        action,
        modified_reward,
        next_obs["grid"],
        next_obs["num_others"],
        next_obs["nav_info"],
        terminated,
        truncated
    )

    # E. Update State and Cache Action for the Next Time Step
    obs = next_obs
    prev_action = action.copy()  # Cache current action as t-1 reference
    step_count += 1

    # F. Active Network Optimization Trigger
    if global_step >= warmup_steps and global_step % update_frequency == 0:
        update_networks(batch_size, global_step)

    # G. Episode Termination & Environment Resets
    if done:
        print(f"Step {global_step} | Episode ended after {step_count} steps.")
        obs, info = env.reset()
        step_count = 0
        # Reset the action tracker to prevent carrying transition penalties across episodes
        prev_action = np.zeros(2, dtype=np.float32)

    # H. Periodic Checkpoint Saving
    if (global_step + 1) % 10000 == 0:
        checkpoint = {
            "global_step": global_step,
            "cnn_extractor": cnn_extractor.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        }
        torch.save(checkpoint, f"checkpoints/checkpoint_final_{global_step}.pt")
        print(f"Checkpoint successfully saved at step {global_step}")

# =====================================================================
# 9. Final Robust Checkpoint Saving
# =====================================================================
final_checkpoint = {
    "global_step": global_step,
    "cnn_extractor": cnn_extractor.state_dict(),
    "actor": actor.state_dict(),
    "critic": critic.state_dict(),
}
torch.save(final_checkpoint, f"checkpoints/checkpoint_final_{global_step}.pt")
print(f"Final checkpoint saved at step {global_step}")

env.close()
print("="*60)
print("TRAINING PIPELINE SUCCESSFULLY TERMINATED")
print("="*60)
