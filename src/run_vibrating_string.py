import numpy as np

from vibrating_string import VibratingString
from visualisation import generate_animation


def initial_disp(x, L, N):
    return np.sin(2*np.pi*x)

def initial_disp2(x, L, N):
    return np.sin(5*np.pi*x)

def initial_disp3(x, L, N):
    if (0.2*L < x*N < 0.4*L):
        return np.sin(5*np.pi*x)
    else:
        return 0


def run_sim(L, T, N, c):
    displacement_func = initial_disp
    vib_string = VibratingString(displacement_func, L, T, N, c)
    vib_string.u = vib_string.compute_next_u(vib_string.u)
    generate_animation(vib_string)

if __name__ == "__main__":
    L = 1.0          
    N = 1000         
    c = 1.0          
    T = 1.0          
    run_sim(L, T, N, c)