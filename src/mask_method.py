import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def check_in_mask(i, j, dx, mask_center, mask_dims, mask_type):
    """
    Checks if a point is inside the mask. Returns a boolean True if it is.
    """
    if mask_type == 'rectangle':
        x, y = i * dx, j * dx
        mask_x, mask_y = mask_center
        mask_width, mask_height = mask_dims
        
        return (abs(x - mask_x) <= mask_width / 2 and 
                abs(y - mask_y) <= mask_height / 2)

    return False


def get_next_grid_mask(grid, N, dt, D, dx, method, omega, mask_center,
    mask_dims, mask_type):
    """
    Performs one iteration of the SOR method to update the grid while 
    respecting the mask.

    Parameters:
    grid (array): Represents the grid
    N (int): Grid size (assumes square NxN grid)
    dt (float): Time step (not used directly in SOR)
    D (float) Diffusion coefficient (not used directly in SOR)
    dx (float): Grid spacing
    method (str): Numerical method ('SOR' supported)
    omega (float): Relaxation parameter for SOR
    mask_center (Tuple): Tuple (x_center, y_center) for mask
    mask_dims: Size of the mask (L for square, R for circle)
    mask_type: 'Square' or 'Circle'

    Returns an updated grid after one iteration and the maximum change 
        in grid values (for convergence tracking)
    """
    new_grid = np.copy(grid)
    maximum_difference = 0

    if method == 'SOR':
        # Interior points only
        for i in range(1, N-1):
            # Full range (with periodic BCs in y-direction)
            for j in range(N):

                # Only update if not masked
                if not check_in_mask(i, j, dx, mask_center, mask_dims, mask_type):
                    old_value = grid[i, j]
                    gauss_seidel_value = 0.25 * (new_grid[i-1, j] + grid[i+1, j] 
                        +  new_grid[i, (j-1) % N] + grid[i, (j+1) % N])
                    new_grid[i, j] = ((1 - omega) * old_value + omega 
                        * gauss_seidel_value)
                    maximum_difference = max(maximum_difference, 
                        abs(new_grid[i, j] - old_value))

        return new_grid, maximum_difference


def simulate_diffusion_mask(N, D, dx, dt, T, method, mask_center, mask_dims, mask_type, omega=1.85, tol=1e-5, save_interval=100):
    """
    Performs one iteration of the SOR method to update the grid while 
    respecting the mask.

    Parameters:
    grid (array): Represents the grid
    N (int): Grid size (assumes square NxN grid)
    dt (float): Time step (not used directly in SOR)
    D (float) Diffusion coefficient (not used directly in SOR)
    dx (float): Grid spacing
    method (str): Numerical method ('SOR' supported)
    omega (float): Relaxation parameter for SOR
    mask_center (Tuple): Tuple (x_center, y_center) for mask
    mask_dims: Size of the mask (L for square, R for circle)
    mask_type: 'Square' or 'Circle'

    Returns an updated grid after one iteration and the maximum change in 
        grid values (for convergence tracking)
    """
    c = initialize_grid(N)
    c_history = [c.copy()]
    if method in ['SOR']:
        iters = 0
        max_difference = []

        while True:
            c, max_diff = get_next_grid_mask(c, N, dt, D, dx, method, omega, 
                mask_center, mask_dims, mask_type)
            
            if iters % save_interval == 0:
                c_history.append(c.copy())
                
            if max_diff < tol:
                print(f"Converged after {iters} iterations")
                break

            max_difference.append(max_diff)
            iters += 1    
        
    return iters, c_history, max_difference


def configure_rectangle_mask(N, D, dx, T, dt):
    """
    Configure simulation parameters for a rectangular mask.
    Returns a configuration dictionary containing mask parameters:
        center (tuple): (x, y) coordinates of mask center in physical units.
        dims (tuple): (width, height) of mask in physical units.
        type (str): Type of mask ('rectangle').
    """
    mask_config = {
        'center': (N * dx * 0.75, N * dx / 2),  
        'dims': (10 * dx, 10 * dx),
        'type': 'rectangle'
    }

    return mask_config


#rename to include mask
def run_single_simulation(omega, mask_config, N, D, dx, dt, T, tol):
    """
    Runs a single diffusion simulation using the specified parameters.
    Returns the number of iterations taken to converge (iters), a list of 
    concentration grids at each saved iteration (c_history) and a list of 
    maximum differences at each iteration for convergence tracking (max_diff).
    """
    print(f'\nRunning simulation with mask at {mask_config['center']}, ' 
          f'ω={omega}')
          
    iters, c_history, max_diff = simulate_diffusion_mask(
        N=N, D=D, dx=dx, dt=dt, T=T,
        method='SOR',
        mask_center=mask_config['center'],
        mask_dims=mask_config['dims'],
        mask_type=mask_config['type'],
        omega=omega,
        tol=tol
    )
    return iters, c_history, max_diff


def run_experiments_on_mask(N, D, dx, T, dt):
    """Main function to run the experiments."""

    mask_config = configure_rectangle_mask(N, D, dx, T, dt)

    omega_values = np.linspace(1.5, 1.90, 15)
    # Stores the number of iterations for each omega
    iterations_list = []
    # Stores the results for each omega
    results = {}

    for omega in omega_values:
        iters, c_history, max_diff = run_single_simulation(
            omega, mask_config, N, D, dx, dt, T, tol=1e-5
        )
        
        # Stores the number of iterations for the current omega
        iterations_list.append(iters)
        
        # Stores results for current omega
        results[omega] = {
            'iterations': iters,
            'final_state': c_history[-1],
            'convergence': max_diff
        }
        
        if omega == 1.5:
            plot_concentration_for_omega(omega, c_history, mask_config, dx)

    # Plots omega vs number of iterations
    plot_iterations(omega_values, iterations_list)

    return results


