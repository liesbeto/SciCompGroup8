import numpy as np
import copy
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

from scipy.special import erfc


#introduce tolerance conditions later
def analytic_sol(x, D, t, max_range):
    """
    D: diffusion coefficient
    x: space/place
    t: time at which to evaluate
    max_range: """

    sum_analytical = 0
    for i in range(max_range):
        sum_analytical += erfc((1 - x + 2*i) / (2*np.sqrt(D*t))) - erfc((1 + x + 2 * i) / (2*np.sqrt(D*t)))

    return sum_analytical


def get_init_2Dgrid(N):
    grid = np.zeros((N, N))
    grid[0, :] = 1  
    grid[-1, :] = 0
    return grid

def iterate_through_grid(grid, dt, D, dx):
    N = grid.shape[0]
    new_grid = copy.deepcopy(grid)

    # skip row zero and last
    for i in range(1, N-1):  
        for j in range(N):
            if j not in [0, N-1]:
                new_grid[i][j] = grid[i][j] + (dt * D / dx**2) * (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1] - 4 * grid[i][j])
            
            # check and update according to boundary conditions
            else:
                if j == N-1:
                    new_grid[i][j] = grid[i][j] + (dt * D / dx**2) * (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][0] - 4 * grid[i][j])
                else:
                    new_grid[i][j] = grid[i][j] + (dt * D / dx**2) * (grid[i-1][j] + grid[i+1][j] + grid[i][N-1] + grid[i][j+1] - 4 * grid[i][j])
    
    
    return new_grid


N = 50 # Number of divisions (length/time scale)
L = 10.0 # Length of square
T = 5.0 # Total time
D = 0.1 # Diffusion coefficient (constant)
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