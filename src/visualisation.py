"""This module is used to visualise the code such as vibrating strings 
in animation form"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from vibrating_string import VibratingString
from iteration_methods import analytical_solution
from iteration_methods import simulate_diffusion_2d, get_next_grid, initialize_grid

def create_animation_vibstring(vib_string):
    """Creates an animation of the bibrating string class."""
    fig, ax = plt.subplots()

    ax.set_xlim(0, 1.5*vib_string.L)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Position")
    ax.set_ylabel("Displacement")
    ax.set_title("Wave Propagation Simulation")
    ax.grid(True, linestyle="--", linewidth=0.5)
    
    line, = ax.plot([], [], 'b-', lw=2, label="Wave Motion")
    ax.legend()
    
    def update(frame):
        line.set_data(vib_string.spatial, vib_string.u[:, frame])
        return line,
    
    ani = animation.FuncAnimation(fig, update, frames=vib_string.time_steps, 
        interval=vib_string.dt*1000, blit=True)
    return ani


def vibrating_string_graphs(vib_string, timesteps):
    """timesteps: list of timesteps at which to plot"""

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1.5*vib_string.L)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Position")
    ax.set_ylabel("Displacement")
    ax.set_title("Wave Propagation Simulation")
    ax.grid(True, linestyle="--", linewidth=0.5)
    for item in timesteps:
        ax.plot(vib_string.spatial, vib_string.u[:, item], label=f"time = {item/1000}")

    ax.legend()
    plt.show()

    return fig, ax


def analytical_plot(D, L, scatter_on=False):
    t_values = [0.001, 0.01, 0.1, 1]
    x_values = np.linspace(0, L, 100)

    fig = plt.figure(figsize=(10, 8))

    for t in t_values:
        analytic_sum = analytical_solution(x_values, t, D, max_range=10)
        plt.plot(x_values, analytic_sum, label=f't = {t}')
        if scatter_on:
            plt.scatter(x_values, analytic_sum)

    plt.xlabel('y position')
    plt.ylabel('Concentration c(y)')
    plt.show()

    return fig


def explicit_method_data(dt, D, dx, N, iterations=100):
    """Saves grids generated by the explicit method in a 3D array where 
    each layer represents the next grid."""
    # Store each grid in a list
    grids = []
    
    grid = initialize_grid(N)

    for i in iterations:
        grid = explicit_method(grid, dt, D, dx)
        grids.append(grid)

    # Set grids as array (3D) to save as binary file
    grids_array = np.array(grids)
    np.save(f'results/explicit_method_grids_i{iterations}.npy', grids_array)



def validate_against_analytical(x_points, times, D, c_history, N):
    """
    Validate the numerical solution against the analytical solution.

    Parameters:
    x_points (array): Spatial points where the concentration is evaluated.
    times (array): Time instances at which the numerical solutions are 
        available.
    D (float): Diffusion coefficient.
    c_history (list of array): Concentration fields at each time step.
    N (int): Size of the grid (NxN).
    """
    plt.figure(figsize=(12, 8))
    mid_x = N // 2
    
    for i, t in enumerate(times):
        if t > 0:  
            numerical = c_history[i][:, mid_x]
            analytical = np.array([analytical_solution(x, t, D) 
                for x in x_points])
            
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


def create_animation(times, c_history, N, dx):
    """
    Create an animation of the concentration field over time.

    Parameters:
    times (array): Time steps for the animation.
    c_history (list of array): Concentration fields at each time step.
    N (int): Size of the grid (NxN).
    dx (float): Spatial resolution of the grid.

    Returns an animation of the object.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    x = np.linspace(0, (N-1) * dx, N)
    y = np.linspace(0, (N-1) * dx, N)
    X, Y = np.meshgrid(x, y)
    
    im = ax.pcolormesh(X, Y, c_history[0], cmap='viridis', shading='auto', 
        vmin=0, vmax=1)
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



def plot_2d_concentration(times, c_history, N, dx):
    """
    Plot 2D concentration fields for given time steps.

    Parameters:
    times (array): Time steps at which the concentration fields are 
        evaluated.
    c_history (list of array): Concentration fields at each time step.
    N (int): Size of the grid (NxN).
    dx (float): Spatial resolution of the grid.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    x = np.linspace(0, (N-1) * dx, N)
    y = np.linspace(0, (N-1) * dx, N)
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


def compare_analytic_solutions(N, L, D, dx, dt, T, method='Explicit'):
    """
    Compare the analytical solution with a numerical method to check simulation accuracy.

    Parameters:
    N (int): Number of grid points in one dimension.
    L (float): Length of the domain.
    D (float): Diffusion coefficient.
    dx (float): Spatial resolution of the grid.
    dt (float): Time step for the simulation.
    T (float): Total simulation time.
    method (str): Numerical method to use ('Explicit' is default).

    Returns an animation object.
    """
    time_points, c_history= simulate_diffusion_2d(N, D, dx, dt, T, method, 
        save_interval=100)
    x_points = np.linspace(0, L, N)
    target_times = [0, 0.001, 0.01, 0.1, 1.0]
    selected_indices = []
    selected_times = []

    for target in target_times:
        # Find indices closest to the target timesteps
        idx = np.argmin(np.abs(np.array(time_points) - target))
        selected_indices.append(idx)
        selected_times.append(time_points[idx])

    validate_against_analytical(x_points, [time_points[i] 
        for i in selected_indices], D, [c_history[i] for i in selected_indices], 
        N)
    # plot_2d_concentration([time_points[i] for i in selected_indices], 
    #     [c_history[i] for i in selected_indices], N, dx)
    # ani = create_animation(time_points, c_history, N, dx)
    
    # return ani


def compare_numerical_methods(N, D, dx, dt, T, L, methods=['Jacobi', 
    'Gauss-Seidel', 'SOR'], tol= 1e-5, omega=1.8):
    """
    Compares numerical methods for solving the diffusion equation.
    Plots the concentration profiles for each method against the analytical solution.
    
    Parameters:
    N (int): Number of grid points
    D (float): Diffusion coefficient
    dx (float): Spatial step size
    dt (float): Time step size
    T (float): Final simulation time
    L (float): Length of the domain
    methods (str): List of numerical methods to compare
    omega (float): Relaxation parameter for SOR
    """
    y_values = np.linspace(0, L, N)
    analytic_vals = [float(analytical_solution(y, T, D)) for y in y_values]
    
    plt.figure(figsize=(10, 6))
    
    for method in methods:
        iters, c_history, _ = simulate_diffusion_2d(N, D, dx, dt, T, 
            method=method, tol= tol, omega=omega)
        c_history = np.array(c_history)

        # Midpoint in x-direction
        mid_x = len(c_history[-1]) // 2
        c_values = [row[mid_x] for row in c_history[-1]]
        
        plt.plot(y_values, c_values, label=f'{method} Method', marker='o')
    
    plt.plot(y_values, analytic_vals, label='Analytical Solution', 
        linestyle='--', color='black')

    plt.xlabel('Y Values')
    plt.ylabel('Concentration')
    plt.title(f'Comparison of {method} Method with Analytical Solution' 
              f'after {iters} iterations')
    plt.legend()
    plt.grid()
    plt.show()


def compare_iterative_methods(N, D, dx, dt, T, tol=1e-5, save_interval=1):
    """
    Compare convergence of Jacobi, Gauss-Seidel, and SOR methods
    
    Parameters:
    N (int): Grid size
    D (float): Diffusion coefficient
    dx (float): Grid spacing
    dt (float): ime step
    T (float): Total simulation time
    tol (float): Convergence tolerance
    save_interval (int): Interval for saving convergence data
        
    Returns a plot comparing convergence of different methods
    """
    results = {}

    iters, _, max_diff = simulate_diffusion_2d(N, D, dx, dt, T, method='Jacobi', 
        tol=tol)
    results['Jacobi'] = max_diff

    iters, _, max_diff = simulate_diffusion_2d(N, D, dx, dt, T, 
        method='Gauss-Seidel', tol=tol)
    results['Gauss-Seidel'] = max_diff
    
    omega_values = [1.5, 1.65, 1.8]
    for omega in omega_values:
        iters, _, max_diff = simulate_diffusion_2d(N, D, dx, dt, T, 
            method='SOR', omega=omega, tol=tol)
        results[f'SOR (ω={omega})'] = max_diff
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot each method
    for method, max_diff in results.items():
        iterations = np.arange(len(max_diff))
        plt.plot(iterations, max_diff, label=method)
    
    plt.xlabel('Iteration')
    plt.ylabel('Maximum Difference')
    plt.title('Convergence Comparison of Iterative Methods')
    plt.yscale('log')  
    plt.grid(True)
    plt.legend()
    
   
    plt.savefig('convergence_comparison.png', dpi=300)
    plt.show()  
    
    return plt.gcf()  


def optimal_omega(N, D, dx, dt, T, tol=1e-5, omega_list=None):
    """
    Find the optimal relaxation parameter (ω) for the SOR method based on 
    the number of iterations required for convergence.

    Parameters:
    N (int): Grid size
    D (float): Diffusion coefficient
    dx (float): Grid spacing
    dt (float): Time step
    T (float): Total simulation time
    tol (float): Convergence tolerance (default is 1e-5).
    omega_list (array, optional): Omega values to test (default is a 
        linear space from 1.1 to 1.9).

    Returns a float of the optimal relaxation parameter (ω) that 
        minimizes the number of iterations.
    """
    # Use a default range of omega values if none is provided
    if omega_list is None:
        omega_list = np.linspace(1.1, 1.9, 20)
    
    iterations = []

    for omega in omega_list:
        iters, _ ,_= simulate_diffusion_2d(N, D, dx, dt, T, method='SOR', 
            tol=tol, omega=omega)
        iterations.append(iters)
    
    plt.figure(figsize=(8, 5))
    plt.plot(omega_list, iterations, marker='o', linestyle='-')
    plt.xlabel('ω (Relaxation Parameter)')
    plt.ylabel('Number of Iterations')
    plt.title(f'Convergence Iterations vs. ω for N = {N}')
    plt.grid(True)
    plt.show()
    
    optimal_idx = np.argmin(iterations)
    optimal_value = omega_list[optimal_idx]
    print(f'Optimal ω: {optimal_value} with {iterations[optimal_idx]} '
           'iterations')
    
    return optimal_value


def plot_omega_vs_N(N_values, D, dt, T, tol=1e-5, omega_list=None):
    """Plots grid size (NxN) verses the optimal omega value."""
    if omega_list is None:
        omega_list = np.linspace(1.5, 2.0, 10)  

    optimal_omegas = []  

    plt.figure(figsize=(8, 5))

    for N in N_values:
        iterations = []
        count = 0
        for omega in omega_list:
            iters, _, _ = simulate_diffusion_2d(N, D, 1/N , dt, T, method='SOR', tol=tol, omega=omega)

            if len(iterations) > 1 and iters > 10 * iterations[-1]:
                break
            iterations.append(iters)

        # Find optimal ω for this N
        optimal_idx = np.argmin(iterations)
        optimal_value = omega_list[optimal_idx]
        optimal_omegas.append(optimal_value)

        print(f"For N={N}, Optimal ω: {optimal_value} with {iterations[optimal_idx]} iterations")

    # Plot N vs optimal omega
    plt.figure(figsize=(8, 5))
    plt.plot(N_values, optimal_omegas, marker='o', linestyle='-', color='blue')
    plt.xlabel('Grid Size N')
    plt.ylabel('Optimal ω (Relaxation Parameter)')
    plt.title('Optimal ω vs. Grid Size N')
    plt.grid(True)
    plt.show()

    return optimal_omegas


def plot_concentration(c_grid, mask_center=None, mask_dims=None, dx=1.0, 
    title="Concentration Field"):
    """
    Plot the concentration field with pcolormesh and optional mask overlay.

    Parameters:
    c_grid (array): 2D array of concentration values.
    mask_center (Tuple, optional): (x, y) coordinates of mask center 
        in physical units.
    mask_dims (tuple, optional): (width, height) of mask in physical 
        units.
    dx (float): Grid spacing.
    title (str): Plot title.
    """
    plt.figure(figsize=(10, 8))

    Ny, Nx = c_grid.shape
    x = np.arange(Nx + 1) * dx
    y = np.arange(Ny + 1) * dx
    X, Y = np.meshgrid(x, y)

    plt.pcolormesh(X, Y, c_grid, cmap='viridis', vmin=0, vmax=1, shading='auto')
    plt.colorbar(label='Concentration')

    plt.gca().set_aspect('equal')  
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.show()


#rename include mask
def plot_concentration_for_omega(omega, c_history, mask_config, dx):
    """
    Plot the concentration field for a specific omega value."""
    plot_concentration(c_history[-1], mask_config['center'], mask_config['dims'], 
        dx, f"Final Concentration (ω={omega}, Mask at {mask_config['center']})")


def plot_iterations(omega_values, iterations_list):
    """Plot the number of iterations for each omega value."""
    plt.figure(figsize=(10, 6))
    plt.plot(omega_values, iterations_list, marker='o')
    plt.title('Omega vs Number of Iterations')
    plt.xlabel('Omega (ω)')
    plt.ylabel('Number of Iterations')
    plt.grid(True)

    # Customize x-ticks to show only certain values
    plt.xticks(np.linspace(1.5, 1.9, num=5))  

    plt.show()



def initial_condition1(x, L, N):
    return np.sin(2*np.pi*x)

def initial_condition2(x, L, N):
    return np.sin(5*np.pi*x)

def initial_condition3(x, L, N):
    if (0.2*L < x*N < 0.4*L):
        return np.sin(5*np.pi*x)
    else:
        return 0


def tolerances_comparison(N, D, dx, dt, T, tol_values=None, omega=1.7):
    """
    Compare the number of iterations needed for convergence across different numerical methods
    with varying tolerance values.

    Parameters:
    N (int): Number of grid points in each dimension.
    D (float): Diffusion coefficient.
    dx (float): Grid spacing.
    dt (float): Time step size.
    T (float): Total simulation time.
    tol_values (list, optional): Tolerance values to test 
        (logarithmically spaced if None).
    omega (float, optional): Relaxation parameter for SOR method 
        (default is 1.7).
    """
     # Generate logarithmically spaced tolerance values
    if tol_values is None:
        tol_values = np.logspace(-5, -14, 6)

    methods = ['Jacobi', 'Gauss-Seidel', 'SOR']
    iter_counts = {method: [] for method in methods}

    for method in methods:
        for tol in tol_values:
            iters, _ ,_= simulate_diffusion_2d(N, D, dx, dt, T, method=method, 
                tol=tol, omega=omega)
            iter_counts[method].append(iters)

 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for i, method in enumerate(methods):
        axes[i].plot(tol_values, iter_counts[method], marker='o', linestyle='-')
        axes[i].set_xscale('log')  
        axes[i].set_xlabel('Tolerance')
        axes[i].set_title(f'{method} Method')
        axes[i].grid(True)

    axes[0].set_ylabel('Iterations')
    plt.suptitle('Iterations needed to converge vs. Tolerance for Different '
                 'Methods')
    plt.tight_layout()
    plt.show()