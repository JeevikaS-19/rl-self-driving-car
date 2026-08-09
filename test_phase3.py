import torch
from phase3 import SACActor, SACCritic  # adjust import to your actual filename

actor = SACActor()
critic = SACCritic()

dummy_obs_actor = torch.randn(5, 2048)          # batch of 5, lidar-latent only
dummy_obs_critic = torch.randn(5, 2064)         # batch of 5, lidar-latent + num_others
dummy_action = torch.randn(5, 2)                # batch of 5, steering+throttle

# test actor
a_squashed, log_prob = actor(dummy_obs_actor)
print("Action shape:", a_squashed.shape)         # expect [5, 2]
print("Log prob shape:", log_prob.shape)         # expect [5, 1]
print("Action range check (should be within [-1,1]):", a_squashed.min().item(), a_squashed.max().item())

# test deterministic mode
a_det, log_prob_det = actor(dummy_obs_actor, deterministic=True)
print("Deterministic log_prob (expect None):", log_prob_det)

# test critic
q1, q2 = critic(dummy_obs_critic, dummy_action)
print("Q1 shape:", q1.shape)   # expect [5, 1]
print("Q2 shape:", q2.shape)   # expect [5, 1]