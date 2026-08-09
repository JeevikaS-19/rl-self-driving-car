from metadrive.envs.metadrive_env import MetaDriveEnv
env = MetaDriveEnv(dict(use_render=False))
print(env.config["out_of_road_penalty"])
print(env.config["speed_reward"])
print(env.config["driving_reward"])
print(env.config["crash_vehicle_penalty"])