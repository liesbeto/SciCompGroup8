"""This Module runs the vibrating string by passing an initial condition function."""

import numpy as np

from vibrating_string import VibratingString
from visualisation import generate_animation


def initial_condition1(x, L, N):
    return np.sin(2*np.pi*x)

def initial_condition2(x, L, N):
    return np.sin(5*np.pi*x)

def initial_condition3(x, L, N):
    if (0.2*L < x*N < 0.4*L):
        return np.sin(5*np.pi*x)
    else:
        return 0


def run_sim(L, T, N, c):
    # Create a vibrating string with passed initial condition
    displacement_func = initial_condition1
    vib_string = VibratingString(displacement_func, L, T, N, c)
    vib_string.u = vib_string.stepping_method(vib_string.u)

    # Generate an animation
    generate_animation(vib_string)


if __name__ == "__main__":
    L = 1.0          
    N = 1000         
    c = 1.0          
    T = 1.0          
    run_sim(L, T, N, c)