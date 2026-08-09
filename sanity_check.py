import torch
from phase3 import SACCritic

critic = SACCritic(latent_dim=2048, cheat_dim=16, action_dim=2)
dummy_obs_critic = torch.randn(5, 2064)
dummy_action = torch.randn(5, 2)

q1, q2 = critic(dummy_obs_critic, dummy_action)
print("Q1 shape:", q1.shape)  # expect [5, 1]
print("Q2 shape:", q2.shape)  # expect [5, 1]
print("Q1 values:", q1.squeeze(-1))
print("Q2 values:", q2.squeeze(-1))