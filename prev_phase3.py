import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class SACActor(nn.Module):
    def __init__(self, latent_dim=2048, hidden_dim=256, action_dim=2, log_std_min=-20, log_std_max=2):
        super(SACActor, self).__init__()

        #clamping bounds
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        #compressing
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs_actor, deterministic=False):
        #process the latent state through the chared backbone
        x = F.relu(self.linear1(obs_actor))
        x = F.relu(self.linear2(x))

        #gaussian distribution parameters 
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        std = torch.exp(log_std)

        #reparameterized trick
        normal_dist = Normal(mu, std)

        #deterministic vs stochastic
        if deterministic:
            a_raw = mu
            log_prob = None
        else: 
            a_raw = normal_dist.rsample()
            normal_log_prob = normal_dist.log_prob(a_raw)

        #tanh
        a_squashed = torch.tanh(a_raw)

        #change of variables corrections
        if not deterministic:
            epsilon = 1e-6
            correction = torch.log(1.0 - a_squashed.pow(2)+epsilon)

            log_prob = normal_log_prob - correction
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        return a_squashed, log_prob

class SACCritic(nn.Module):
    def __init__(self, latent_dim=2048, cheat_dim=16, action_dim=2, hidden_dim=256):
        super(SACCritic, self).__init__()

        #total input dimensionality
        #osb_critic = 2048 + 16 = 2064
        #action = steering, throttle
        input_dim = latent_dim + cheat_dim + action_dim

        #critic 1 
        self.q1_layer1 = nn.Linear(input_dim, hidden_dim)
        self.q1_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1) #outputs a single scalar q value

        #critic 2
        self.q2_layer1 = nn.Linear(input_dim, hidden_dim)
        self.q2_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

    def forward(self, obs_critic, action):
        #fusion step
        state_action = torch.cat([obs_critic, action], dim=-1)
        #forward pass
        q1 = F.relu(self.q1_layer1(state_action))
        q1 = F.relu(self.q1_layer2(q1))
        q1_value = self.q1_out(q1)

        q2 = F.relu(self.q2_layer1(state_action))
        q2 = F.relu(self.q2_layer2(q2))
        q2_value = self.q2_out(q2)

        return q1_value, q2_value
