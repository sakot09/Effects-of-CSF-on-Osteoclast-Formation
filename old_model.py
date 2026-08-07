import numpy as np
from scipy.integrate import odeint


def ode_model(y, t, gamma):
    M, O = y

    dMdt = (-gamma * M * O) - (gamma * (M**2))

    dOdt = -dMdt

    return [dMdt, dOdt]


def solve_ode_model(gamma, M0, O0, t):

    y0 = [M0, O0]

    solution = odeint(ode_model, y0, t, args=(gamma,))

    M = solution[:, 0]
    O = solution[:, 1]


    return M, O




