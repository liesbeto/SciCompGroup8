import numpy as np


class VibratingString():
    """
    L: Length of string
    T: Total time of the simulation
    N: Number of timesteps/ Number of divisions of spatial grid
    """
    def __init__(self, displacement_func, L, T, N, c):
        self.N = N
        self.c = c
        self.L = L
        self.temporal_grid, self.dx = self.grid(L)
        self.spatial_grid, self.dt = self.grid(T)
        self.time_steps = int(T / self.dt)
        self.set_initial_disp(displacement_func)
       
        if not self.check_stability(c):
            raise ValueError("cfl number greater than 1")

    def grid(self, endpoint):
        """discretizing time steps (temporal grid) or delta X (spatial grid)"""
        return np.linspace(0, endpoint, self.N+1), endpoint/self.N

    def check_stability(self, c):
        """Returns True if the timestep length is stable according to Courant-Friedrichs-Law."""
        return (c * self.dt / self.dx) <= 1

    def set_initial_disp(self, displacement_func):
        u = np.zeros((self.N+1, self.time_steps+1))
        for i in range(len(u)):
            u[i] = displacement_func(self.temporal_grid[i], len(u), self.N)
        
        u[:, 1] = np.copy(u[:, 0])

        self.u = u

    # Only do once and use loop somewhere else because this is not next this is all?
    # Remove u and use self.u?
    def compute_next_u(self, u):
        c_sq = self.c**2
        for n in range(1, self.time_steps):
            for i in range(1, self.N):
                u[i, n+1] = 2 * u[i, n] - u[i, n-1] + c_sq * (u[i+1, n] - 2*u[i, n] + u[i-1, n])
        return u
