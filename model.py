import numpy as np
from scipy.integrate import odeint


def ode_model(y, t, gamma):
    

    dydt = np.zeros(50)

    dydt[0] = -gamma * (y[0]**2)

    total = np.sum(y)

    for i in range(1, 50):
        positive = 0
        for j in range(i):
            positive += y[j] * y[i-j-1]
        dydt[i] = gamma * positive - gamma * y[i] * total
    
    return dydt
        
        


def solve_ode_model(gamma, y0, t):

    t_dense = np.linspace(0, max(t), 1000)
    solution_dense = odeint(ode_model, y0, t_dense, args=(gamma,), mxstep=5000)
    
    indices = [np.argmin(np.abs(t_dense - time)) for time in t]
    solution = solution_dense[indices, :]
    
    return solution




