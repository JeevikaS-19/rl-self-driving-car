import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class SACActor(nn.Module):
    def __init__(self, latent_dim=2058, hidden_dim=256, action_dim=2, log_std_min=-20, log_std_max=2):
        super(SACActor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs_actor, deterministic=False):
        x = F.relu(self.linear1(obs_actor))
        x = F.relu(self.linear2(x))

        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        normal_dist = Normal(mu, std)

        if deterministic:
            a_raw = mu
            log_prob = None
        else:
            a_raw = normal_dist.rsample()
            normal_log_prob = normal_dist.log_prob(a_raw)

        a_squashed = torch.tanh(a_raw)

        if not deterministic:
            epsilon = 1e-6
            correction = torch.log(1.0 - a_squashed.pow(2) + epsilon)
            log_prob = normal_log_prob - correction
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        return a_squashed, log_prob

class SACCritic(nn.Module):
    def __init__(self, latent_dim=2058, cheat_dim=16, action_dim=2):
        super().__init__()

        self.state_dim = latent_dim + cheat_dim
        self.action_dim = action_dim

        # TWIN CRITIC 1
        self.v1_linear1 = nn.Linear(self.state_dim, 256)
        self.v1_linear2 = nn.Linear(256, 128)
        self.v1_out = nn.Linear(128, 1)

        self.a1_linear1 = nn.Linear(self.state_dim + self.action_dim, 256)
        self.a1_linear2 = nn.Linear(256, 128)
        self.a1_out = nn.Linear(128, 1)

        # TWIN CRITIC 2
        self.v2_linear1 = nn.Linear(self.state_dim, 256)
        self.v2_linear2 = nn.Linear(256, 128)
        self.v2_out = nn.Linear(128, 1)

        self.a2_linear1 = nn.Linear(self.state_dim + self.action_dim, 256)
        self.a2_linear2 = nn.Linear(256, 128)
        self.a2_out = nn.Linear(128, 1)

    def forward(self, obs_critic, action):
        """
        Args:
            obs_critic: [Batch, 2064]
            action: [Batch, 2]
        Returns:
            q1, q2: [Batch, 1]
        """
        # --- Stream 1 ---
        v1 = F.relu(self.v1_linear1(obs_critic))
        v1 = F.relu(self.v1_linear2(v1))
        v1_value = self.v1_out(v1)

         # We use F.leaky_relu for the Advantage Stream A1
        # to guarantee the backpropagation path for action gradients remains open.
        state_action = torch.cat([obs_critic, action], dim=-1)
        a1 = F.leaky_relu(self.a1_linear1(state_action), negative_slope=0.01)
        a1 = F.leaky_relu(self.a1_linear2(a1), negative_slope=0.01)
        a1_value = self.a1_out(a1)
        q1_value = v1_value + a1_value

        # --- Stream 2 ---
        v2 = F.relu(self.v2_linear1(obs_critic))
        v2 = F.relu(self.v2_linear2(v2))
        v2_value = self.v2_out(v2)

        a2 = F.leaky_relu(self.a2_linear1(state_action), negative_slope=0.01)
        a2 = F.leaky_relu(self.a2_linear2(a2), negative_slope=0.01)
        a2_value = self.a2_out(a2)

        q2_value = v2_value + a2_value

        return q1_value, q2_value