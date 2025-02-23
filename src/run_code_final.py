import numpy as np
import time
import copy
import random
import typing

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from scipy.special import erfc


N = 50                  # Number of grid points in each dimension (50x50 grid)
L = 1.0                 # Length of the domain (1.0 unit)
dx = L / N              # Grid spacing (distance between grid points)
D = 1.0                 # Diffusion coefficient (controls the rate of diffusion)
dt = 0.25 * dx**2 / D   # Time step size based on stability criteria
T = 1.0                 # Total simulation time
c = 1.0
timesteps = [0, 100, 250, 500, 700]


from vibrating_string import VibratingString
from visualisation import create_animation_vibstring, vibrating_string_graphs


def initial_condition1(x, L, N):
    return np.sin(2*np.pi*x)

def initial_condition2(x, L, N):
    return np.sin(5*np.pi*x)

def initial_condition3(x, L, N):
    if (0.2*L < x*N < 0.4*L):
        return np.sin(5*np.pi*x)
    else:
        return 0

def assignment1a():
    # # Create a vibrating string with passed initial condition
    displacement_func = initial_condition1
    vib_string1 = VibratingString(displacement_func, L, T, N, c)
    vib_string1.u = vib_string1.stepping_method(vib_string1.u)
    fig, ax = vibrating_string_graphs(vib_string1, timesteps)

    displacement_func = initial_condition2
    vib_string2 = VibratingString(displacement_func, L, T, N, c)
    vib_string2.u = vib_string2.stepping_method(vib_string2.u)
    fig, ax = vibrating_string_graphs(vib_string2, timesteps)

    displacement_func = initial_condition3
    vib_string3 = VibratingString(displacement_func, L, T, N, c)
    vib_string3.u = vib_string3.stepping_method(vib_string3.u)
    fig, ax = vibrating_string_graphs(vib_string3, timesteps)
