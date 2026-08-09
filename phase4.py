import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity=50000, grid_shape=(4,64,64), num_others_dim=16, nav_info_dim=10, action_dim=2):
        self.capacity = capacity
        self.pointer = 0
        self.size = 0

        self.obs_grid_buffer = np.zeros((capacity,) + grid_shape, dtype=np.float32)
        self.obs_num_others_buffer = np.zeros((capacity, num_others_dim), dtype=np.float32)
        self.obs_nav_info_buffer = np.zeros((capacity, nav_info_dim), dtype=np.float32)
        self.action_buffer = np.zeros((capacity, action_dim), dtype=np.float32)

        self.reward_buffer = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs_grid_buffer = np.zeros((capacity,) + grid_shape, dtype=np.float32)
        self.next_obs_num_others_buffer = np.zeros((capacity, num_others_dim), dtype=np.float32)
        self.next_obs_nav_info_buffer = np.zeros((capacity, nav_info_dim), dtype=np.float32)
        self.done_buffer = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs_grid, obs_num_others, obs_nav_info, action, reward,
            next_obs_grid, next_obs_num_others, next_obs_nav_info, terminated, truncated):
        done_mask = 1.0 if terminated else 0.0

        self.obs_grid_buffer[self.pointer] = obs_grid
        self.obs_num_others_buffer[self.pointer] = obs_num_others
        self.obs_nav_info_buffer[self.pointer] = obs_nav_info
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.next_obs_grid_buffer[self.pointer] = next_obs_grid
        self.next_obs_num_others_buffer[self.pointer] = next_obs_num_others
        self.next_obs_nav_info_buffer[self.pointer] = next_obs_nav_info
        self.done_buffer[self.pointer] = done_mask

        self.pointer = (self.pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size=256):
        indices = np.random.randint(0, self.size, size=batch_size)

        batch_grid = torch.FloatTensor(self.obs_grid_buffer[indices])
        batch_num_others = torch.FloatTensor(self.obs_num_others_buffer[indices])
        batch_nav_info = torch.FloatTensor(self.obs_nav_info_buffer[indices])
        batch_actions = torch.FloatTensor(self.action_buffer[indices])
        batch_rewards = torch.FloatTensor(self.reward_buffer[indices])
        batch_next_grid = torch.FloatTensor(self.next_obs_grid_buffer[indices])
        batch_next_num_others = torch.FloatTensor(self.next_obs_num_others_buffer[indices])
        batch_next_nav_info = torch.FloatTensor(self.next_obs_nav_info_buffer[indices])
        batch_dones = torch.FloatTensor(self.done_buffer[indices])

        return (batch_grid, batch_num_others, batch_nav_info, batch_actions, batch_rewards,
                batch_next_grid, batch_next_num_others, batch_next_nav_info, batch_dones)