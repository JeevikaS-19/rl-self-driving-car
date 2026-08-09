import math
import numpy as np
from config import GRID_SIZE, MAX_RANGE, RESOLUTION, EGO_COL, EGO_ROW

def projection_lidar_to_bev(lidar_array, max_range=MAX_RANGE, grid_size=GRID_SIZE):
    resolution = max_range/grid_size #0.78125m/cell
    num_rays = len(lidar_array) # Typically 240
    bev_grid = np.zeros((grid_size,grid_size), dtype=np.float32)
    angular_step = (2*math.pi)/num_rays

    for i in range(num_rays):
        normalized_distance = lidar_array[i]
        if normalized_distance>=0.99:
            continue
        physical_distance = normalized_distance*max_range
        theta = -i*angular_step
        x_meters = physical_distance*math.cos(theta)
        y_meters = physical_distance*math.sin(theta)

        row = EGO_ROW - math.floor(x_meters/resolution)
        col = EGO_COL + math.floor(y_meters / resolution)
        
        if 0<=row<grid_size and 0<=col<grid_size:
            bev_grid[row][col] = 1.0

    return bev_grid

if __name__ == "__main__":
    # fake lidar: all "nothing detected"
    empty = np.ones(240)
    grid = projection_lidar_to_bev(empty)
    print("Empty scene, occupied cells:", grid.sum())  # should print 0.0

    # fake lidar: one ray straight ahead (index 0) detecting something close
    test = np.ones(240)
    test[0] = 0.2  # 0.2 * 50m = 10m directly ahead
    grid = projection_lidar_to_bev(test)
    print("One-point scene, occupied cells:", grid.sum())  # should print 1.0
    print("Occupied at:", np.argwhere(grid == 1.0))