import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Define the analytical solution
def analytical_solution(x, t, D, max_range=10):
    sum_analytical = np.zeros_like(x)
    for i in range(max_range):
        term1 = erfc((1 - x + 2 * i) / (2*np.sqrt(D*t)))
        term2 = erfc((1 + x + 2 * i) / (2*np.sqrt(D*t)))
        sum_analytical += term1 - term2
    return sum_analytical

# Define the grid initialization
def initialize_grid(N):
    grid = np.zeros((N, N))  # Initialize the grid with zeros
    grid[-1, :] = 1.0  # Set the boundary at the bottom (i.e., grid[N-1, :] = 1.0)
    return grid

# Explicit method simulation
def explicit_method(grid, dt, D, dx):
    def grid_center(grid, i, j, dt, dx, D, N):
        return grid[i, j] + (dt * D / dx**2) * (grid[i-1, j] + grid[i+1, j] + grid[i, j-1] + grid[i, j+1] - 4 * grid[i, j])

    def grid_boundary_right(grid, i, j, dt, dx, D, N):
        return grid[i, j] + (dt * D / dx**2) * (grid[i-1, j] + grid[i+1, j] + grid[i, j-1] + grid[i, 0] - 4 * grid[i, j])

    def grid_boundary_left(grid, i, j, dt, dx, D, N):
        return grid[i, j] + (dt * D / dx**2) * (grid[i-1, j] + grid[i+1, j] + grid[i, j+1] + grid[i, N-1] - 4 * grid[i, j])

    N = grid.shape[0]
    new_grid = grid.copy()

    # Apply updates for the interior and boundary points
    for i in range(1, N-1):
        for j in range(N):
            if j == 0:
                new_grid[i, j] = grid_boundary_left(grid, i, j, dt, dx, D, N)
            elif j == N - 1:
                new_grid[i, j] = grid_boundary_right(grid, i, j, dt, dx, D, N)
            else:
                new_grid[i, j] = grid_center(grid, i, j, dt, dx, D, N)

    return new_grid

# Main simulation and comparison
def run_simulation(N, D, dx, dt, max_time, plot_times):
    # Initialize grid
    grid = initialize_grid(N)

    # Space grid (x values)
    x = np.linspace(0, N*dx, N)
    
    # Time steps (t values)
    times = np.arange(0, max_time, dt)

    # Set up a figure for plotting
    plt.figure(figsize=(10, 6))

    # Run the simulation and compare at different times
    for t in plot_times:
        # Run the simulation to get the solution at time `t`
        num_steps = int(t / dt)
        for _ in range(num_steps):
            grid = explicit_method(grid, dt, D, dx)
        
        # Analytical solution at time `t`
        analytical = analytical_solution(x, t, D)

        # We need to plot the concentration at the middle row of the grid (this represents c(y))
        # Get the row representing the concentration at a fixed position, for example, the middle row
        numerical_solution = grid[int(N/2), :]  # Center row of the grid

        # Plot the numerical solution and the analytical solution for comparison
        plt.plot(x, numerical_solution, label=f'Numerical, t={t:.2f}')
        plt.plot(x, analytical, label=f'Analytical, t={t:.2f}', linestyle='--')

    plt.xlabel('Space (x)')
    plt.ylabel('Concentration (c)')
    plt.legend()
    plt.title('Comparison of Numerical and Analytical Solutions')
    plt.grid(True)
    plt.show()

# Parameters
N = 100  # Grid size
D = 1.0  # Diffusion coefficient
dx = 0.1  # Space step
dt = 0.01  # Time step
max_time = 1.0  # Max time for simulation
plot_times = [0.01, 0.1, 1.0]  # Times to plot the comparison

# Run the simulation
run_simulation(N, D, dx, dt, max_time, plot_times)
