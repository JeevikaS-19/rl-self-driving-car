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
os.makedirs("checkpoints", exist_ok=True)

# 1. Initialize the Environment with the 16-dim num_others activated
config = dict(
    use_render=False,             
    out_of_road_penalty=15.0,
    crash_vehicle_penalty=15.0,
    use_lateral_reward=True,
    vehicle_config=dict(lidar=dict(num_lasers=240, distance=50, num_others=4))
)
raw_env = MetaDriveEnv(config)
env = FrameStackWrapper(raw_env)

# 2. Instantiate the Memory Buffer
replay_buffer = ReplayBuffer(capacity=200000)

# 3. Instantiate the Neural Network Modules
cnn_extractor = CNNAttentionExtractor()
actor = SACActor(latent_dim=2048, action_dim=2)
critic = SACCritic(latent_dim=2048, cheat_dim=16, action_dim=2)

# 4. Instantiate Target Networks for Polyak Averaging
target_cnn_extractor = copy.deepcopy(cnn_extractor)
target_critic = copy.deepcopy(critic)

#learnable entropy temp
target_entropy = -2.0
log_alpha = torch.tensor(np.log(0.3), requires_grad=True, dtype= torch.float32)

# 5. Setup Optimizers
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
cnn_optimizer = torch.optim.Adam(cnn_extractor.parameters(), lr=3e-4)
alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4)

# 6. Hyperparameters
max_steps = 15000
warmup_steps = 5000
update_frequency = 4
batch_size = 256
gamma = 0.90
tau = 0.005




# 7. The optimization engine — MUST be defined before the loop that calls it
def update_networks(batch_size, global_step):
    (batch_grid, batch_num_others, batch_actions, batch_rewards,batch_next_grid, batch_next_num_others, batch_dones) = replay_buffer.sample(batch_size)
    alpha = log_alpha.exp()
    #compute sac bellman target
    with torch.no_grad():
        next_latent = target_cnn_extractor(batch_next_grid)
        next_action, next_log_prob = actor(next_latent, deterministic=False)
        next_obs_critic = torch.cat([next_latent, batch_next_num_others], dim=-1)
        target_q1, target_q2 = target_critic(next_obs_critic, next_action)
        min_target_q = torch.min(target_q1, target_q2)
        target_q = batch_rewards + gamma * (1.0 - batch_dones) * (min_target_q - alpha * next_log_prob)
    #update the critic
    latent = cnn_extractor(batch_grid)
    obs_critic = torch.cat([latent, batch_num_others], dim=-1)
    current_q1, current_q2 = critic(obs_critic, batch_actions)
    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

    critic_optimizer.zero_grad()
    cnn_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    cnn_optimizer.step()

    #update the actor

    detached_latent = latent.detach()
    fresh_action, fresh_log_prob = actor(detached_latent, deterministic=False)
    fresh_obs_critic = torch.cat([detached_latent, batch_num_others], dim=-1)
    actor_q1, actor_q2 = critic(fresh_obs_critic, fresh_action)
    actor_q = torch.min(actor_q1, actor_q2)
    actor_loss = (alpha * fresh_log_prob - actor_q).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Automatic entropy temperature update
    alpha_loss = -(log_alpha * (fresh_log_prob.detach() + target_entropy)).mean()

    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    if global_step % 1000 == 0:
        with torch.no_grad():
            x = torch.relu(actor.linear1(detached_latent))
            x = torch.relu(actor.linear2(x))
            log_std_check = torch.clamp(actor.log_std_head(x), actor.log_std_min, actor.log_std_max)
            std_check = torch.exp(log_std_check)
            print(f"step={global_step} mean_std={std_check.mean().item():.4f} alpha={alpha.item():.4f}")

    if global_step % 1000 == 0:
        with torch.no_grad():
            x = torch.relu(actor.linear1(detached_latent))
            x = torch.relu(actor.linear2(x))
            log_std_check = torch.clamp(actor.log_std_head(x), actor.log_std_min, actor.log_std_max)
            std_check = torch.exp(log_std_check)
            print(f"step={global_step} mean_std={std_check.mean().item():.4f}")

    with torch.no_grad():
        for target_param, live_param in zip(target_critic.parameters(), critic.parameters()):
            target_param.data.copy_(tau * live_param.data + (1.0 - tau) * target_param.data)
        for target_param, live_param in zip(target_cnn_extractor.parameters(), cnn_extractor.parameters()):
            target_param.data.copy_(tau * live_param.data + (1.0 - tau) * target_param.data)


#training loop (this must exist separately, below this function)
obs, info = env.reset()
#comfort and smoothness penalty //additional --> step 8
step_count = 0

# Initialize previous action tracker to zeros to begin smoothness penalty tracking
prev_action = np.zeros(2, dtype=np.float32)

print("\n" + "="*60)
print("LAUNCHING TRAINING PIPELINE")
print("="*60)

for global_step in range(max_steps):
    # A. Action Selection: Warmup Phase vs. Active Stochastic Policy
    if global_step < warmup_steps:
        # Uniform sampling to populate experience replay with high-entropy transitions
        action = raw_env.action_space.sample()
    else:
        grid_tensor = torch.FloatTensor(obs["grid"]).unsqueeze(0)
        with torch.no_grad():
            latent = cnn_extractor(grid_tensor)
            # Sample stochastically during training to maintain exploration pressure
            action_tensor, _ = actor(latent, deterministic=False)
            action = action_tensor.squeeze(0).numpy()

    # B. Execute Action in MetaDrive Environment
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # C. STEP B: Calculate Action-Smoothness (Comfort) Penalties
    # Penalize rapid, high-frequency counter-steering and violent throttle/brake jumps [8]
    # action is steering, action[1] is throttle/brake
    steer_change = (action - prev_action) ** 2
    throt_change = (action[1] - prev_action[1]) ** 2
    
    # Scale comfort penalties (standard control multiplier w_comfort = -0.1) [11]
    comfort_penalty = -0.1 * steer_change - 0.1 * throt_change
    modified_reward = reward + comfort_penalty

    # D. Store Grounded, Modified Transition in Replay Buffer
    # We store the modified reward so the Critic learns to value smooth trajectories [9]
    replay_buffer.add(
        obs["grid"],
        obs["num_others"],
        action,
        modified_reward,
        next_obs["grid"],
        next_obs["num_others"],
        done
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


for global_step in range(max_steps):

    # Action Selection Phase
    if global_step < warmup_steps:
        action = raw_env.action_space.sample()  # fixed: raw_env, not the wrapper
    else:
        grid_array = obs["grid"]
        grid_tensor = torch.FloatTensor(grid_array).unsqueeze(0)

        with torch.no_grad():
            latent_vector = cnn_extractor(grid_tensor)
            action_tensor, _ = actor(latent_vector, deterministic=False)
            action = action_tensor.squeeze(0).cpu().numpy()

    # Environment Interaction Phase
    next_obs, reward, terminated, truncated, info = env.step(action)

    # Replay Buffer Storage Phase
    speed = raw_env.agent.speed
    action_is_braking = action[1] < 0.0

    if speed < 0.1 and action_is_braking:
        store_transition = (np.random.rand() < 0.02)
    else:
        store_transition = True

    if store_transition:
        replay_buffer.add(
            obs_grid=obs["grid"],
            obs_num_others=obs["num_others"],
            action=action,
            reward=reward,
            next_obs_grid=next_obs["grid"],
            next_obs_num_others=next_obs["num_others"],
            terminated=terminated,
            truncated=truncated
        )

    obs = next_obs

    # Episode Management
    if terminated or truncated:
        obs, info = env.reset()

    # Checkpoint saving
    if global_step % 50000 == 0 and global_step > 0:
        checkpoint = {
            "global_step": global_step,
            "cnn_extractor": cnn_extractor.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        }
        torch.save(checkpoint, f"checkpoints/checkpoint_{global_step}.pt")
        print(f"step={global_step} checkpoint saved")    

    # Network Update Trigger
    if global_step >= warmup_steps and global_step % update_frequency == 0:
        update_networks(batch_size, global_step)
        print(f"step={global_step} update_networks() ran successfully")

    if global_step % 10 == 0:
        print(f"step={global_step} buffer_size={replay_buffer.size} last_reward={reward:.3f}")

# Final save, regardless of checkpoint interval — ensures short validation runs still produce a usable checkpoint
final_checkpoint = {
    "global_step": global_step,
    "cnn_extractor": cnn_extractor.state_dict(),
    "actor": actor.state_dict(),
    "critic": critic.state_dict(),
}
torch.save(final_checkpoint, f"checkpoints/checkpoint_final_{global_step}.pt")
print(f"Final checkpoint saved at step {global_step}")
