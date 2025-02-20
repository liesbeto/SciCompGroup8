"""This module contains a class for Vibrating String according to the ... method."""

import numpy as np

# TO DO: Implement Eulers method to compare with stepping

class VibratingString():
    """
    Boundaries are assumed fixed 
    L: Length of string
    T: Total time of the simulation
    N: Number of timesteps/ Number of divisions of spatial grid
    """
    def __init__(self, displacement_func, L, T, N, c):
        self.N = N
        self.c = c
        self.L = L
        self.spatial, self.dx = self.boundary(L)
        self.temporal, self.dt = self.boundary(T)
        self.time_steps = int(T / self.dt)
        self.set_initial_disp(displacement_func)
       
        if not self.check_stability(c):
            raise ValueError("cfl number greater than 1")


    def boundary(self, endpoint):
        """discretizing time steps (temporal) or delta X (spatial)"""
        return np.linspace(0, endpoint, self.N+1), endpoint/self.N


    def check_stability(self, c):
        """Returns True if the timestep length is stable according to Courant-Friedrichs-Law."""
        return (c * self.dt / self.dx) <= 1


    def set_initial_disp(self, displacement_func):
        u = np.zeros((self.N+1, self.time_steps+1))
        for i in range(len(u)):
            u[i] = displacement_func(self.spatial[i], len(u), self.N)
        
        u[:, 1] = np.copy(u[:, 0]) 
        self.u = u


    def stepping_method(self, u):
        """Computes u according to stepping method by using two most recent time points to calculate next"""
        c_sq = self.c**2
        for n in range(1, self.time_steps):
            for i in range(1, self.N):
                u[i, n+1] = 2 * u[i, n] - u[i, n-1] + c_sq * (u[i+1, n] - 2*u[i, n] + u[i-1, n])
        return u
