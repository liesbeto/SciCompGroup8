import numpy as np
import matplotlib.pyplot as plt

from vibrating_string import VibratingString

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

def initial_condition1(x, L, N):
    return np.sin(2*np.pi*x)

def initial_condition2(x, L, N):
    return np.sin(5*np.pi*x)

def initial_condition3(x, L, N):
    if (0.2*L < x*N < 0.4*L):
        return np.sin(5*np.pi*x)
    else:
        return 0

def run_vibrating_string(L, T, N, c):
    # Create a vibrating string with passed initial condition
    displacement_func = initial_condition1
    vib_string = VibratingString(displacement_func, L, T, N, c)
    vib_string.u = vib_string.stepping_method(vib_string.u)

    # Generate an animation
    # ani = generate_animation(vib_string)

    # Generate a figure
    fig, ax = vibrating_string_graphs(vib_string, timesteps = [0, 100, 250, 500, 700])


if __name__ == "__main__":
    L = 1.0
    N = 1000
    c = 1.0
    T = 1.0
    run_vibrating_string(L, T, N, c)

    # Plot analytical solution
    # fig = analytical_plot(D=1, L=1)