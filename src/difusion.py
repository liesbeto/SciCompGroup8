import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from scipy.special import erfc
import time
import copy

# We assume square dimensions!
# We assume 0 ≤ x, y ≤ 1.
# Boundary conditions: c(x, y = 1; t) = 1 and c(x, y = 0; t) = 0

#introduce tolerance conditions later
def analytical_solution(x, t, D, max_range=10):
    """
    D: diffusion coefficient
    x: space/place
    t: time at which to evaluate
    max_range: """

    sum_analytical = np.zeros_like(x)
    for i in range(max_range):
        sum_analytical += erfc((1 - x + 2 * i) / (2*np.sqrt(D*t))) - erfc((1 + x + 2 * i) / (2*np.sqrt(D*t)))

    return sum_analytical


def initialize_grid(N):
    grid = np.zeros((N, N))

    grid[-1, :] = 1.0
    
    return grid

# Still work in progress
def boundary_grid(grid, dt, D, dx):

    N = grid.shape[0]
    new_grid = copy.deepcopy(grid)

    # Center of grid
    for i in range(1, N-1):
        for j in range(1, N-1):
            new_grid[i][j] = grid[i][j] + (dt * D / dx**2) * (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1] - 4 * grid[i][j])

        # Boundary where j=0
        new_grid[i][0] = grid[i][0] + (dt * D / dx**2) * (grid[i-1][0] + grid[i+1][0] + grid[i][1] + grid[i][N-1] - 4 * grid[i][0])
        # Boundary where j=N-1
        new_grid[i][N-1] = grid[i][N-1] + (dt * D / dx**2) * (grid[i-1][N-1] + grid[i+1][N-1] + grid[i][N-2] + grid[i][0] - 4 * grid[i][N-1])
    
    # Boundary of grid where i=0 and i=N-1
    for j in range(1, N-1):
        new_grid[0][j] = grid[0][j] + (dt * D / dx**2) * (grid[1][j] + grid[N-1][j] + grid[0][j-1] + grid[0][j+1] - 4 * grid[0][j])
        new_grid[N-1][j] = grid[N-1][j] + (dt * D / dx**2) * (grid[0][j] + grid[N-2][j] + grid[N-1][j-1] + grid[N-1][j+1] - 4 * grid[N-1][j])

    return


def get_next_grid(grid, dt, D, dx):

    N = grid.shape[0]
    new_grid = grid.copy()

    for i in range(1, N-1):
        for j in range(N):
            if j not in [0, N-1]:
                new_grid[i][j] = grid[i][j] + (dt * D / dx**2) * (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][j+1] - 4 * grid[i][j])
            else:
                if j == N-1:
                    new_grid[i][j] = grid[i][j] + (dt * D / dx**2) * (grid[i-1][j] + grid[i+1][j] + grid[i][j-1] + grid[i][0] - 4 * grid[i][j])
                else:
                    new_grid[i][j] = grid[i][j] + (dt * D / dx**2) * (grid[i-1][j] + grid[i+1][j] + grid[i][N-1] + grid[i][j+1] - 4 * grid[i][j])
    return new_grid


def get_next_grid_method(grid, dt, D, dx, method="Jacobi", omega=1.5):
    """Compute next time step using Jacobi, Gauss-Seidel, or SOR method."""
    N = grid.shape[0]
    new_grid = grid.copy()

    if method == "Jacobi":
        old_grid = grid.copy()
        for i in range(1, N - 1):
            for j in range(N):
                if j == 0:
                    new_grid[i, j] = old_grid[i, j] + (dt * D / dx**2) * (old_grid[i-1, j] + old_grid[i+1, j] + old_grid[i, N-1] + old_grid[i, j+1] - 4 * old_grid[i, j])
                elif j == N - 1:
                    new_grid[i, j] = old_grid[i, j] + (dt * D / dx**2) * (old_grid[i-1, j] + old_grid[i+1, j] + old_grid[i, j-1] + old_grid[i, 0] - 4 * old_grid[i, j])
                else:
                    new_grid[i, j] = old_grid[i, j] + (dt * D / dx**2) * (old_grid[i-1, j] + old_grid[i+1, j] + old_grid[i, j-1] + old_grid[i, j+1] - 4 * old_grid[i, j])

    elif method in ["Gauss-Seidel", "SOR"]:
        for i in range(1, N - 1):
            for j in range(N):
                if j == 0:
                    value = grid[i, j] + (dt * D / dx**2) * (grid[i-1, j] + grid[i+1, j] + grid[i, N-1] + grid[i, j+1] - 4 * grid[i, j])
                elif j == N - 1:
                    value = grid[i, j] + (dt * D / dx**2) * (grid[i-1, j] + grid[i+1, j] + grid[i, j-1] + grid[i, 0] - 4 * grid[i, j])
                else:
                    value = grid[i, j] + (dt * D / dx**2) * (grid[i-1, j] + grid[i+1, j] + grid[i, j-1] + grid[i, j+1] - 4 * grid[i, j])
                if method == "SOR":
                    grid[i, j] = (1 - omega) * grid[i, j] + omega * value
                else:
                    grid[i, j] = value
        return grid  # Gauss-Seidel and SOR update in-place

    return new_grid  # Jacobi returns a new array


def simulate_diffusion_2d(N, D, dx, dt, T, save_interval=100):
    # Check stability condition
    stability_param = 4 * D * dt / (dx * dx)
    if stability_param > 1:
        dt = 0.95 * dx**2 / (4 * D)
  
    c = initialize_grid(N)
    
 
    n_steps = int(T / dt)
    time_points = [0.0]
    c_history = [c.copy()]
    
    print(f"Starting simulation with {n_steps} time steps")

    
    for step in range(1, n_steps+1):
        c = get_next_grid(c, dt, D, dx)
        current_time = step * dt

        special_times = [0.001, 0.01, 0.1, 1.0]
        if (step % save_interval == 0) or any(abs(current_time - t) < dt for t in special_times):
            time_points.append(current_time)
            c_history.append(c.copy())


def simulate_diffusion_2d_methods(N, D, dx, dt, T, method="Jacobi", omega=1.5, save_interval=100):
    """Simulate diffusion using Jacobi, Gauss-Seidel, or SOR."""
    stability_param = 4 * D * dt / (dx * dx)
    if stability_param > 1:
        dt = 0.95 * dx**2 / (4 * D)
    
    c = initialize_grid(N)
    n_steps = int(T / dt)
    time_points = [0.0]
    c_history = [c.copy()]
    
    print(f"Running {method} method for {n_steps} steps")

    for step in range(1, n_steps + 1):
        c = get_next_grid(c, dt, D, dx, method=method, omega=omega)
        current_time = step * dt
        special_times = [0.001, 0.01, 0.1, 1.0]

        if (step % save_interval == 0) or any(abs(current_time - t) < dt for t in special_times):
            time_points.append(current_time)
            c_history.append(c.copy())

    return time_points, c_history
            


    return time_points, c_history

def validate_against_analytical(x_points, times, D, c_history, N):

    plt.figure(figsize=(12, 8))
    
    # Get middle x-index
    mid_x = N // 2
    
    for i, t in enumerate(times):
        if t > 0:  

            numerical = c_history[i][:, mid_x]
            
            # Calculate analytical solution
            analytical = np.array([analytical_solution(x, t, D) for x in x_points])
            
            # Plot both solutions
            plt.plot(x_points, numerical, 'o-', label=f'Numerical, t={t:.3f}')
            plt.plot(x_points, analytical, '--', label=f'Analytical, t={t:.3f}')
    
    plt.xlabel('y position')
    plt.ylabel('Concentration c(y)')
    plt.title('Comparison between numerical and analytical solutions')
    plt.legend()
    plt.grid(True)
    plt.savefig('validation_plot.png', dpi=300)
    plt.show()

def validate_methods(y_points, times, D, c_history, N):
    """Compare numerical solutions to analytical steady-state c(y) = y."""
    plt.figure(figsize=(12, 8))
    mid_x = N // 2  # Middle of x-axis

    for i, t in enumerate(times):
        if t > 0:
            numerical = c_history[i][:, mid_x]  # Extract values along x
            analytical = np.linspace(0, 1, N)  # Expected c(y) = y
            plt.plot(y_points, numerical, 'o-', label=f'Numerical, t={t:.3f}')
            plt.plot(y_points, analytical, '--', label=f'Analytical (c=y), t={t:.3f}')

    plt.xlabel('y position')
    plt.ylabel('Concentration c(y)')
    plt.title('Comparison between Numerical and Analytical Solution (c=y)')
    plt.legend()
    plt.grid(True)
    plt.savefig('steady_state_validation.png', dpi=300)
    plt.show()

def plot_2d_concentration(times, c_history, N, dx):

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    x = np.linspace(0, (N-1)*dx, N)
    y = np.linspace(0, (N-1)*dx, N)
    X, Y = np.meshgrid(x, y)
    
    for i, (t, c) in enumerate(zip(times, c_history)):
        if i < len(axes):
            ax = axes[i]
            im = ax.pcolormesh(X, Y, c, cmap='viridis', shading='auto')
            ax.set_title(f't = {t:.3f}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            fig.colorbar(im, ax=ax, label='Concentration')
    
    plt.tight_layout()
    plt.savefig('concentration_plots.png', dpi=300)
    plt.show()

def create_animation(times, c_history, N, dx):
    """
    Create an animation of the concentration field over time
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    x = np.linspace(0, (N-1)*dx, N)
    y = np.linspace(0, (N-1)*dx, N)
    X, Y = np.meshgrid(x, y)
    
    # Initial plot
    im = ax.pcolormesh(X, Y, c_history[0], cmap='viridis', shading='auto', vmin=0, vmax=1)
    title = ax.set_title(f't = {times[0]:.5f}')
    fig.colorbar(im, ax=ax, label='Concentration')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    def update(frame):
        im.set_array(c_history[frame].ravel())
        title.set_text(f't = {times[frame]:.5f}')
        return [im, title]
    
    ani = FuncAnimation(fig, update, frames=range(len(times)), blit=True)
    ani.save('diffusion_animation.gif', writer='pillow', fps=15)
    plt.show()
    
    return ani


# # Simulation parameters
# N= 50      # Number of grid points (must be square grid for provided update function)
# L = 1.0   # Domain size
# dx= L/(N-1)  # Grid spacing
# D = 1.0              # Diffusion coefficient

# # Stability-limited time step
# dt = 0.24 * dx**2 / D  
# T = 1.0   
    
# print(f"Grid: {N}x{N}, dx={dx:.5f}, dt={dt:.6f}")
# print(f"Stability parameter: {4*D*dt/dx**2:.5f} (must be ≤ 1)")
    
# # Run simulation
# time_points, c_history = simulate_diffusion_2d( N, D, dx, dt, T, save_interval=100)

# x_points = np.linspace(0, L, N)


# target_times = [0, 0.001, 0.01, 0.1, 1.0]
# selected_indices = []
# selected_times = []

# for target in target_times:
#     idx = np.argmin(np.abs(np.array(time_points) - target))#Finding indices which are closest to the target timesteps
#     selected_indices.append(idx)
#     selected_times.append(time_points[idx])


# validate_against_analytical(x_points, [time_points[i] for i in selected_indices], D, [c_history[i] for i in selected_indices], N)


# plot_2d_concentration([time_points[i] for i in selected_indices],[c_history[i] for i in selected_indices], N, dx)

# ani = create_animation(time_points, c_history, N, dx)

# methods = ["Jacobi", "Gauss-Seidel", "SOR"]
# omega = 1.5  # Relaxation factor for SOR

# for method in methods:
#     time_points, c_history = simulate_diffusion_2d(N, D, dx, dt, T, method=method, omega=omega)

#     y_points = np.linspace(0, L, N)
#     selected_indices = np.linspace(0, len(time_points) - 1, 5, dtype=int)
#     selected_times = [time_points[i] for i in selected_indices]
#     selected_c_history = [c_history[i] for i in selected_indices]

#     validate_methods(y_points, selected_times, D, selected_c_history, N)
