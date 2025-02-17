'''This module is used to visualise the code such as vibrating strings 
in animation form'''

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from vibrating_string import VibratingString


def generate_animation(vib_string):
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
    
    ani = animation.FuncAnimation(fig, update, frames=vib_string.time_steps, interval=vib_string.dt*1000, blit=True)
    plt.show()

    # Uncomment to save file, may take some time
    # Animation very slow, check later
    # ani.save('results/vibrating_string_animation.gif')


def vibrating_string_graphs(n_graphs):
    """n_graphs: Number of graphs to plot."""

    fig, ax = plt.subplots()
    ax.set_xlim(0, 1.5*vib_string.L)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Position")
    ax.set_ylabel("Displacement")
    ax.set_title("Wave Propagation Simulation")
    ax.grid(True, linestyle="--", linewidth=0.5)
    
    line, = ax.plot([], [], 'b-', lw=2, label="Wave Motion")
    ax.legend()

    return