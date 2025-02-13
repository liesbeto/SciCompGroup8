import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class VibratingString():
    """
    L: Length of string
    T: Total time of the simulation
    N: Number of timesteps/ Number of divisions of spatial grid
    """
    def __init__(self, L, T, N, c):
        self.temporal_grid, self.dx = self.grid(L, N)
        self.spatial_grid, self.dt = self.grid(T, N)
        self.time_steps = int(T / self.dt)
        self.N = N
        self.c = c
        self.L = L
       
        if not self.check_stability(c):
            raise ValueError("cfl number greater than 1")

    def grid(self, endpoint, N):
        """discretizing time steps (temporal grid) or delta X (spatial grid)"""
        return np.linspace(0, endpoint, N+1), endpoint/N

    def check_stability(self, c):
        """Returns True if the timestep length is stable according to Courant-Friedrichs-Law."""
        return (c * self.dt / self.dx) <= 1

    def set_initial_disp(self, u, displacement_func):
        for i in range(len(u)):
            u[i] = displacement_func(self.temporal_grid[i], len(u), self.N)
        return u

    def compute_next_u(self, u):
        c_sq = self.c**2
        for n in range(1, self.time_steps):
            for i in range(1, self.N):
                u[i, n+1] = 2 * u[i, n] - u[i, n-1] + c_sq * (u[i+1, n] - 2*u[i, n] + u[i-1, n])
        return u



def initial_disp(x, L, N):
    return np.sin(2*np.pi*x)

def initial_disp2(x, L, N):
    return np.sin(5*np.pi*x)

def initial_disp3(x, L, N):
    if (0.2*L < x*N < 0.4*L):
        return np.sin(5*np.pi*x)
    else:
        return 0


def generate_animation(vib_string, u):
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
        line.set_data(vib_string.temporal_grid, u[:, frame])
        return line,
    
    ani = animation.FuncAnimation(fig, update, frames=vib_string.time_steps, interval=vib_string.dt*1000, blit=True)
    plt.show()
    

def run_sim(L, T, N, c):
    vib_string = VibratingString(L, T, N, c)
    displacement_func = initial_disp
    
    u = np.zeros((N+1, vib_string.time_steps+1))
    u[:, 0] = vib_string.set_initial_disp(u[:, 0], displacement_func)
    u[:, 1] = np.copy(u[:, 0])

    u = vib_string.compute_next_u(u)

    generate_animation(vib_string, u)


L = 1.0          
N = 1000         
c = 1.0          
T = 1.0          
run_sim(L, T, N, c)
