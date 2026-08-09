import torch
from phase3 import SACActor

checkpoint = torch.load("checkpoints/checkpoint_final_14999.pt")
actor = SACActor(latent_dim=2048, action_dim=2)
actor.load_state_dict(checkpoint["actor"])
actor.eval()

for i in range(5):
    fake_latent = torch.randn(1, 2048) * 5  # varied random input
    with torch.no_grad():
        action, _ = actor(fake_latent, deterministic=True)
    print(f"random_input_{i}: action={action.squeeze(0).numpy()}")