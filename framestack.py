from collections import deque
import numpy as np
import torch
from phase1 import projection_lidar_to_bev
from config import GRID_SIZE, MAX_RANGE


class FrameStackWrapper:
    def __init__(self, env, num_frames=4, max_range=MAX_RANGE, grid_size=GRID_SIZE):
        self.env = env
        self.num_frames = num_frames
        self.max_range = max_range
        self.grid_size = grid_size
        self.frames = deque(maxlen=num_frames)

    def _stacked_tensor(self):
        stacked = np.stack(list(self.frames), axis=0)
        return torch.from_numpy(stacked).float()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        lidar_slice = obs[35:275]
        num_others = obs[19:35]
        nav_info = obs[9:19]
        grid = projection_lidar_to_bev(lidar_slice, max_range=self.max_range, grid_size=self.grid_size)

        self.frames.clear()
        for _ in range(self.num_frames):
            self.frames.append(grid)

        return {
            "grid": self._stacked_tensor(),
            "num_others": torch.from_numpy(num_others.copy()).float(),
            "nav_info": torch.from_numpy(nav_info.copy()).float(),
        }, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        lidar_slice = obs[35:275]
        num_others = obs[19:35]
        nav_info = obs[9:19]
        grid = projection_lidar_to_bev(lidar_slice, max_range=self.max_range, grid_size=self.grid_size)

        self.frames.append(grid)

        return {
            "grid": self._stacked_tensor(),
            "num_others": torch.from_numpy(num_others.copy()).float(),
            "nav_info": torch.from_numpy(nav_info.copy()).float(),
        }, reward, terminated, truncated, info

    def close(self):
        self.env.close()