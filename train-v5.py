import torch
import torch.nn.functional as F
import numpy as np
import copy
import pickle
from metadrive.envs.metadrive_env import MetaDriveEnv
from framestack import FrameStackWrapper
from perception import CNNAttentionExtractor
from phase3 import SACActor, SACCritic
from phase4 import ReplayBuffer
import os

os.makedirs("checkpoints", exist_ok=True)

log_path = "checkpoints/training_logs_v4.csv"
if not os.path.exists(log_path):
    with open(log_path, "w") as f:
        f.write("step,critic_loss,actor_loss,entropy_alpha,step_reward\n")

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

replay_buffer = ReplayBuffer(capacity=200000)


def load_demonstrations(replay_buffer, demo_path="demonstrations.pkl"):
    with open(demo_path, "rb") as f:
        demos = pickle.load(f)

    for transition in demos:
        replay_buffer.add(
            transition["obs_grid"],
            transition["obs_num_others"],
            transition["obs_nav_info"],
            transition["action"],
            transition["reward"],
            transition["next_obs_grid"],
            transition["next_obs_num_others"],
            transition["next_obs_nav_info"],
            transition["terminated"],
            transition["truncated"],
        )
    print(f"Loaded {len(demos)} demonstration transitions into replay buffer.")


load_demonstrations(replay_buffer)

cnn_extractor = CNNAttentionExtractor()
actor = SACActor(latent_dim=2058, action_dim=2)
critic = SACCritic(latent_dim=2058, cheat_dim=16, action_dim=2)

target_cnn_extractor = copy.deepcopy(cnn_extractor)
target_critic = copy.deepcopy(critic)

target_entropy = -2.0
log_alpha = torch.tensor(np.log(0.3), requires_grad=True, dtype=torch.float32)

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
cnn_optimizer = torch.optim.Adam(cnn_extractor.parameters(), lr=3e-4)
alpha_optimizer = torch.optim.Adam([log_alpha], lr=3e-4)

max_steps = 750000  #i think we have it, finally, last ver i hope 
warmup_steps = 5000
update_frequency = 4
batch_size = 256
gamma = 0.90
tau = 0.005
#i forgot to put nav_scale 
nav_scale = 0.1


def update_networks(batch_size, global_step):
    #LOCATON B --> FORGOT TO DEBUG HERE AS WELL (UGHHH) --> now updated
    (batch_grid, batch_num_others, batch_nav_info, batch_actions, batch_rewards, batch_next_grid, batch_next_num_others, batch_next_nav_info, batch_dones) = replay_buffer.sample(batch_size)

    alpha = log_alpha.exp()
    scaled_batch_nav_info = batch_nav_info * nav_scale
    scaled_batch_next_nav_info = batch_next_nav_info * nav_scale

    with torch.no_grad():
        next_latent_raw = target_cnn_extractor(batch_next_grid)
        next_actor_input = torch.cat([next_latent_raw, scaled_batch_next_nav_info], dim=-1)
        next_action, next_log_prob = actor(next_actor_input, deterministic=False)

        next_obs_critic = torch.cat([next_latent_raw, scaled_batch_next_nav_info, batch_next_num_others], dim=-1)
        target_q1, target_q2 = target_critic(next_obs_critic, next_action)
        min_target_q = torch.min(target_q1, target_q2)

        target_q = batch_rewards + gamma * (1.0 - batch_dones) * (min_target_q - alpha * next_log_prob)
    #location D
    latent_raw = cnn_extractor(batch_grid)

    obs_critic = torch.cat([latent_raw, scaled_batch_nav_info, batch_num_others], dim=-1)

    current_q1, current_q2 = critic(obs_critic, batch_actions)
    critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

    critic_optimizer.zero_grad()
    cnn_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    cnn_optimizer.step()

    #LOCATION C --> now updated
    detached_latent = latent_raw.detach()
    #apply scale current nav info in mini-batch
    detached_actor_input = torch.cat([detached_latent, scaled_batch_nav_info], dim=-1)
    #HAHAH LET ME FIX THIS TOO 
    fresh_action, fresh_log_prob = actor(detached_actor_input, deterministic=False)

    fresh_obs_critic = torch.cat([detached_latent, scaled_batch_nav_info, batch_num_others], dim=-1)
    actor_q1, actor_q2 = critic(fresh_obs_critic, fresh_action)
    actor_q = torch.min(actor_q1, actor_q2)

    actor_loss = (alpha * fresh_log_prob - actor_q).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    alpha_loss = -(log_alpha.exp() * (fresh_log_prob.detach() + target_entropy)).mean()

    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    with torch.no_grad():
        for target_param, live_param in zip(target_critic.parameters(), critic.parameters()):
            target_param.data.copy_(tau * live_param.data + (1.0 - tau) * target_param.data)
        for target_param, live_param in zip(target_cnn_extractor.parameters(), cnn_extractor.parameters()):
            target_param.data.copy_(tau * live_param.data + (1.0 - tau) * target_param.data)

    return critic_loss.item(), actor_loss.item(), alpha.item()


obs, info = env.reset()
step_count = 0
prev_action = np.zeros(2, dtype=np.float32)

print("\n" + "="*60)
print("LAUNCHING TRAINING PIPELINE (v5 - with num scenarios updated)")
print("="*60)

for global_step in range(max_steps):
    if global_step < warmup_steps:
        action = raw_env.action_space.sample()
    else:
        #LOCATION A (now updated)
        grid_tensor = torch.FloatTensor(obs["grid"]).unsqueeze(0)
        nav_tensor = torch.FloatTensor(obs["nav_info"]).unsqueeze(0)
        with torch.no_grad():
            latent = cnn_extractor(grid_tensor)
            #my dumbass forgot to actually update this, no wonder it aint working
            #appling scale down navigation coordinates dynamically
            scaled_nav_tensor = nav_tensor*nav_scale
            actor_input = torch.cat([latent, scaled_nav_tensor], dim=-1)

            action_tensor, _ = actor(actor_input, deterministic=False)
            action = action_tensor.squeeze(0).numpy()

    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    steer_change = (action[0] - prev_action[0]) ** 2
    throt_change = (action[1] - prev_action[1]) ** 2
    comfort_penalty = 0.0
    modified_reward = reward + comfort_penalty

    speed = raw_env.agent.speed
    action_is_braking = action[1] < 0.0
    store_transition = (np.random.rand() < 0.02) if (speed < 0.1 and action_is_braking) else True

    if store_transition:
        replay_buffer.add(
            obs["grid"], obs["num_others"], obs["nav_info"],
            action, modified_reward,
            next_obs["grid"], next_obs["num_others"], next_obs["nav_info"],
            terminated, truncated
        )

    obs = next_obs
    prev_action = action.copy()
    step_count += 1

    if global_step >= warmup_steps and global_step % update_frequency == 0:
        c_loss, a_loss, alpha_val = update_networks(batch_size, global_step)
        with open(log_path, "a") as f:
            f.write(f"{global_step},{c_loss:.6f},{a_loss:.6f},{alpha_val:.6f},{reward:.6f}\n")

    if done:
        print(f"Step {global_step} | Episode ended after {step_count} steps.")
        obs, info = env.reset()
        step_count = 0
        prev_action = np.zeros(2, dtype=np.float32)

    if (global_step + 1) % 10000 == 0:
        checkpoint = {
            "global_step": global_step,
            "cnn_extractor": cnn_extractor.state_dict(),
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        }
        torch.save(checkpoint, f"checkpoints/checkpoint_v6_{global_step}.pt")
        print(f"Checkpoint saved at step {global_step}")

final_checkpoint = {
    "global_step": global_step,
    "cnn_extractor": cnn_extractor.state_dict(),
    "actor": actor.state_dict(),
    "critic": critic.state_dict(),
}
torch.save(final_checkpoint, f"checkpoints/checkpoint_v6_final_{global_step}.pt")
print(f"Final checkpoint saved at step {global_step}")

env.close()
print("="*60)
print("TRAINING PIPELINE SUCCESSFULLY TERMINATED")
print("="*60)