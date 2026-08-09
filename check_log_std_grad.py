import torch
from phase3 import SACActor

# Fresh, untrained actor — same initialization as any new run
fresh_actor = SACActor(latent_dim=2048, action_dim=2)

# Load your most recent trained checkpoint
checkpoint = torch.load("checkpoints/checkpoint_final_49999.pt")  # adjust to your latest checkpoint filename
trained_actor = SACActor(latent_dim=2048, action_dim=2)
trained_actor.load_state_dict(checkpoint["actor"])

# Compare log_std_head weights: fresh vs trained
fresh_weights = fresh_actor.log_std_head.weight
trained_weights = trained_actor.log_std_head.weight

diff = (fresh_weights - trained_weights).abs()
print("log_std_head weight difference — mean:", diff.mean().item())
print("log_std_head weight difference — max:", diff.max().item())

# Compare against mu_head for contrast — we know mu is learning fine
fresh_mu = fresh_actor.mu_head.weight
trained_mu = trained_actor.mu_head.weight
diff_mu = (fresh_mu - trained_mu).abs()
print("mu_head weight difference — mean:", diff_mu.mean().item())
print("mu_head weight difference — max:", diff_mu.max().item())