import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.special import erfc
def initialize_grid(N: int):
    """
    Initialize a 2D grid of size N x N with the bottom row set to 1.0.
    """
    grid = np.zeros((N, N))
    grid[-1, :] = 1.0
    
    return grid


def get_next_grid(grid, N, dt, D, dx, method, omega):
    """
    Compute the next iteration of the 2D grid based on the specified 
    method.

    Parameters:
    grid (array): The current state of the grid.
    N (int): Size of the grid (number of rows and columns).
    dt (float): Time step for the simulation.
    D (float): Diffusion coefficient.
    dx (float): Spatial step size.
    method (str): The numerical method to use ('Explicit', 'Jacobi', 
        'Gauss-Seidel', or 'SOR').
    omega (float): Relaxation factor for the SOR method.

    Returns (tuple):
        - new_grid (array): The updated grid after applying the numerical 
            method.
        - maximum_difference (float, optional): The maximum difference 
            between the old and new grid values (only for Jacobi, 
            Gauss-Seidel, and SOR methods).
    """
    new_grid = np.copy(grid)

    if method == 'Explicit':
        for i in range(1, N - 1):
            for j in range(N):
                new_grid[i, j] = grid[i, j] + (dt * D / dx**2) * (
                    grid[i - 1, j] + grid[i + 1, j] + grid[i, (j - 1) % N] 
                    + grid[i, (j + 1) % N] - 4 * grid[i, j])
                    
        return new_grid, None

    elif method == 'Jacobi':
        maximum_difference = 0
        for i in range(1, N - 1):
            for j in range(N):
                new_grid[i, j] = 0.25 * (
                    grid[i - 1, j] + grid[i + 1, j] + grid[i, (j - 1) % N] 
                    + grid[i, (j + 1) % N])

                maximum_difference = max(maximum_difference, abs(new_grid[i, j] 
                    - grid[i, j]))

        return new_grid, maximum_difference  

    elif method == 'Gauss-Seidel':
        maximum_difference = 0
        for i in range(1, N - 1):
            for j in range(N):
                old_value = new_grid[i, j]
                new_grid[i, j] = 0.25 * (
                    new_grid[i - 1, j] + grid[i + 1, j] 
                        + new_grid[i, (j - 1) % N] + grid[i, (j + 1) % N]
                )
                maximum_difference = max(maximum_difference, 
                    abs(new_grid[i, j] - old_value))

        return new_grid, maximum_difference  

    elif method == 'SOR':
        maximum_difference = 0
        for i in range(1, N - 1):
            for j in range(N):
                old_value = grid[i, j]

                gauss_seidel_value = 0.25 * (
                    new_grid[i - 1, j] + grid[i + 1, j] 
                    + new_grid[i, (j - 1) % N] + grid[i, (j + 1) % N])

                new_grid[i, j] = ((1 - omega) * old_value 
                    + omega * gauss_seidel_value)

                maximum_difference = max(maximum_difference, 
                    abs(new_grid[i, j] - old_value))

        return new_grid, maximum_difference


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


def simulate_diffusion_2d(N, D, dx, dt, T, method="Explicit", omega=1.85, tol=1e-5, save_interval=100):
    """
    Simulates the process of diffusion through a 2D grid.

    Parameters:
    N (int): The size of the grid (N x N).
    D (float): The diffusion coefficient.
    dx (float): The grid spacing.
    dt (float): The time step for the simulation.
    T (float): The total time for the simulation.
    method (str, optional): The numerical method for diffusion 
        ('Explicit', 'Jacobi', 'Gauss-Seidel', 'SOR'). 
        Default is 'Explicit'.
    omega (float, optional): The relaxation parameter for SOR 
        (if applicable). Default is 1.85.
    tol (float, optional): The tolerance for convergence 
        (for iterative methods). Default is 1e-5.
    save_interval (int, optional): The interval at which the grid state 
        is saved. Default is 100.

    Returns a list of time points at which the grid states are saved and
        a list of the grid states over time.
    
    (For iterative methods)
    iters (int): The number of iterations taken to converge.
    max_difference (list): Maximum differences between successive 
        iterations.
    """
    c = initialize_grid(N)
    c_history = [c.copy()]

    n_steps = int(T / dt)
    time_points = [0.0]
    special_times = [0.001, 0.01, 0.1, 1.0]

    if method == 'Explicit':
        for step in range(1, n_steps + 1):
            c,_= get_next_grid(c, N, dt, D, dx, method=method, omega=omega)
            current_time = step * dt

            if (step % save_interval == 0) or any(abs(current_time - t) 
                < dt for t in special_times):

                time_points.append(current_time)
                c_history.append(c.copy())

        return time_points, c_history
    
    elif method in ['Jacobi', 'Gauss-Seidel', 'SOR']:
        iters = 0
        max_difference = []
        while True:  
            c, max_diff = get_next_grid(c, N, dt, D, dx, method=method, 
                omega=omega)
            
            if iters % save_interval == 0:
                c_history.append(c.copy())
                
            if max_diff < tol:
                print(f"Converged after {iters} iterations")
                break

            max_difference.append(max_diff)
            iters += 1
            
        return iters, c_history, max_difference   