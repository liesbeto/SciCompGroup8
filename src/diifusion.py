import numpy as np
import copy
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

def get_init_2Dgrid(N):
    grid = np.zeros((N, N))
    grid[0, :] = 1  
    grid[-1, :] = 0
    return grid

def iterate_through_grid(grid, dt, D, dx):
    N = grid.shape[0]
    new_grid = copy.deepcopy(grid)

    for i in range(1, N-1):  
        for j in range(N):
            if j not in [0,N-1]:
                new_grid[i][j] = grid[i][j] + (dt * D / dx**2) * (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1] - 4 * grid[i][j])
            else:
                if j == N-1:
                    new_grid[i][j] = grid[i][j] + (dt * D / dx**2) * (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][0] - 4 * grid[i][j])
                else:
                    new_grid[i][j] = grid[i][j] + (dt * D / dx**2) * (grid[i-1][j] + grid[i+1][j] + grid[i][N-1] + grid[i][j+1] - 4 * grid[i][j])
    
    
    return new_grid

N = 50 
L = 10.0 
T= 5.0
D = 0.1 
max_iters = 200  

interval = 5


dx= L/N
dt = dx**2 / (4 * D)

grid = get_init_2Dgrid(N)


fig, ax = plt.subplots()
heatmap = ax.imshow(grid, cmap="hot", interpolation="nearest", vmin=0, vmax=1)
plt.colorbar(heatmap)


def update(frame):
    global grid
    grid = iterate_through_grid(grid, dt, D, dx)
    heatmap.set_array(grid)
    return [heatmap]


ani = animation.FuncAnimation(fig, update, frames=max_iters, interval=interval, blit=False)


plt.title("2D Diffusion Simulation")
plt.show()