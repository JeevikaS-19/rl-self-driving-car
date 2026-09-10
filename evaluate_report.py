import torch
import numpy as np
import json
from metadrive.envs.metadrive_env import MetaDriveEnv
from framestack import FrameStackWrapper
from perception import CNNAttentionExtractor
from phase3 import SACActor

checkpoint_path = "checkpoints/checkpoint_v7_final_999999.pt"  # adjust to your exact 750k filename
NUM_EPISODES = 10
NAV_SCALE = 0.1
MAX_STEPS_PER_EPISODE = 2000

print(f"Loading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

cnn_extractor = CNNAttentionExtractor()
actor = SACActor(latent_dim=2058, action_dim=2)
cnn_extractor.load_state_dict(checkpoint["cnn_extractor"])
actor.load_state_dict(checkpoint["actor"])
cnn_extractor.eval()
actor.eval()

config = dict(
    use_render=False,
    num_scenarios=200,
    out_of_road_penalty=15.0,
    crash_vehicle_penalty=15.0,
    use_lateral_reward=True,
    vehicle_config=dict(lidar=dict(num_lasers=240, distance=50, num_others=4))
)
raw_env = MetaDriveEnv(config)
env = FrameStackWrapper(raw_env)

episode_reports = []

for ep in range(NUM_EPISODES):
    obs, info = env.reset()
    step_count = 0
    max_speed = 0.0
    min_speed_after_moving = None
    braking_events = 0
    was_braking_last_step = False
    reached_dest = False
    scenario_index = info.get("env_seed", "unknown")

    for step in range(MAX_STEPS_PER_EPISODE):
        grid_tensor = torch.FloatTensor(obs["grid"]).unsqueeze(0)
        nav_tensor = torch.FloatTensor(obs["nav_info"]).unsqueeze(0)
        with torch.no_grad():
            latent = cnn_extractor(grid_tensor)
            scaled_nav = nav_tensor * NAV_SCALE
            actor_input = torch.cat([latent, scaled_nav], dim=-1)
            action, _ = actor(actor_input, deterministic=True)
        action_np = action.squeeze(0).numpy()

        obs, reward, terminated, truncated, info = env.step(action_np)
        step_count += 1
        speed = raw_env.agent.speed
        max_speed = max(max_speed, speed)

        is_braking_now = action_np[1] < -0.1 and speed > 1.0
        if is_braking_now and not was_braking_last_step:
            braking_events += 1
        was_braking_last_step = is_braking_now

        if info.get("arrive_dest", False):
            reached_dest = True

        if terminated or truncated:
            break

    fail_reason = "arrive_dest" if reached_dest else info.get("out_of_road", False) and "out_of_road" or info.get("crash", False) and "crash" or "max_steps_reached"

    episode_reports.append({
        "episode": ep + 1,
        "scenario_index": scenario_index,
        "steps_survived": step_count,
        "max_speed": round(max_speed, 2),
        "braking_events": braking_events,
        "reached_destination": reached_dest,
        "end_reason": fail_reason,
        "route_completion": round(float(info.get("route_completion", 0.0)), 4),
    })

    print(f"Episode {ep+1}/{NUM_EPISODES} | steps={step_count} | max_speed={max_speed:.2f} | "
          f"braking_events={braking_events} | route_completion={info.get('route_completion', 0):.3f} | "
          f"end_reason={fail_reason}")

raw_env.close()

# Summary
steps_list = [r["steps_survived"] for r in episode_reports]
completion_list = [r["route_completion"] for r in episode_reports]
dest_count = sum(1 for r in episode_reports if r["reached_destination"])

summary = {
    "checkpoint": checkpoint_path,
    "num_episodes": NUM_EPISODES,
    "mean_steps_survived": round(float(np.mean(steps_list)), 1),
    "median_steps_survived": round(float(np.median(steps_list)), 1),
    "min_steps_survived": int(np.min(steps_list)),
    "max_steps_survived": int(np.max(steps_list)),
    "mean_route_completion": round(float(np.mean(completion_list)), 4),
    "episodes_reaching_destination": dest_count,
    "episodes_with_braking_events": sum(1 for r in episode_reports if r["braking_events"] > 0),
    "episodes": episode_reports,
}

with open("evaluation_report_v7.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(f"Episodes run: {NUM_EPISODES}")
print(f"Mean steps survived: {summary['mean_steps_survived']}")
print(f"Median steps survived: {summary['median_steps_survived']}")
print(f"Range: {summary['min_steps_survived']} - {summary['max_steps_survived']}")
print(f"Mean route completion: {summary['mean_route_completion']*100:.1f}%")
print(f"Episodes reaching destination: {dest_count}/{NUM_EPISODES}")
print(f"Episodes with at least one braking event: {summary['episodes_with_braking_events']}/{NUM_EPISODES}")
print(f"Full report saved to evaluation_report.json")
print("=" * 60)