import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class VibratingString():
    def __init__():
        self.temporal_grid = temporal_grid(L, N)
        self.spatial_grid = spatial_grid(T, N)

    def temporal_grid(L, N):
        """discretizing time steps"""
        return np.linspace(0, L, N+1), L/N

    def spatial_grid(T, N):
        """discretizing delta X"""
        return np.linspace(0, T, N+1), T/N

    return

def temporal_grid(L, N):
    """discretizing time steps"""
    return np.linspace(0, L, N+1), L/N

def spatial_grid(T, M):
    """discretizing delta X"""
    return np.linspace(0, T, M+1), T/M

def check_stability(dt, dx, c):
    """Courant-Friedrichs-Law or CFL conditions, how to choose time step size
    Returns True if stable (lower than or equal to 1). If time step too small ..."""
    return (c * dt / dx) <= 1

def initial_disp(x, i, L, N):
    return np.sin(2*np.pi*x)

def initial_disp2(x, i, L, N):
    return np.sin(5*np.pi*x)

def initial_disp3(x, i, L, N):
    if (0.2*L < x*N < 0.4*L):
        return np.sin(5*np.pi*x)
    else:
        return 0


def set_initial_disp(u, x):
    for i in range(len(u)):
        u[i] = initial_disp2(x[i], i, len(u), N)
    return u

def compute_next_u(u, c, N, time_steps):
    c_sq = c**2
    for n in range(1, time_steps):
        for i in range(1, N):
            u[i, n+1] = 2 * u[i, n] - u[i, n-1] + c_sq * (u[i+1, n] - 2*u[i, n] + u[i-1, n])
    return u

def generate_animation(x, u, L, time_steps, dt):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1.5*L)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("Position")
    ax.set_ylabel("Displacement")
    ax.set_title("Wave Propagation Simulation")
    ax.grid(True, linestyle="--", linewidth=0.5)
    
    line, = ax.plot([], [], 'b-', lw=2, label="Wave Motion")
    ax.legend()
    
    def update(frame):
        line.set_data(x, u[:, frame])
        return line,
    
    ani = animation.FuncAnimation(fig, update, frames=time_steps, interval=dt*1000, blit=True)
    plt.show()
    

def run_sim(L, T, N, c):
    x, dx = temp_grid(L, N)
    _, dt = spatial_grid(T, N)
    time_steps = int(T / dt)

    if not check_stability(dt, dx, c):
        raise ValueError("cfl number greater than 1")
    
    u = np.zeros((N+1, time_steps+1))
    u[:, 0] = set_initial_disp(u[:, 0], x)
    u[:, 1] = np.copy(u[:, 0])
    u = compute_next_u(u, c, N, time_steps)
    
    generate_animation(x, u, L, time_steps, dt)


L = 1.0          
N = 1000         
c = 1.0          
T = 1.0          
run_sim(L, T, N, c)
