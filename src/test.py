
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

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


def run_vibrating_string(L, T, N, c):
    # Create a vibrating string with passed initial condition
    displacement_func = initial_condition1
    vib_string = VibratingString(displacement_func, L, T, N, c)
    #vib_string.u = vib_string.stepping_method(vib_string.u)

    # Generate an animation
    #ani = generate_animation(vib_string)

    # Generate a figure
    fig, ax = vibrating_string_graphs(vib_string, timesteps = [0, 0.1, 0.25, 0.5, 0.7])

def initial_condition1(x, L, N):
    return np.sin(2*np.pi*x)

run_vibrating_string(L=1, T=1, N=50, c=1)