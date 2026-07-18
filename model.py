import numpy as np
from scipy.integrate import solve_ivp


<<<<<<< HEAD
def ivp_model(y, t, gamma):
=======
def ivp_model(t, y, gamma):
>>>>>>> b8cda375a62a31263e566330b6724178c3afc139
    

    dydt = np.zeros(50)

    dydt[0] = -gamma * (y[0]**2)

    total = np.sum(y)

    for i in range(1, 50):
        positive = 0
        for j in range(i):
            positive += y[j] * y[i-j-1]
        dydt[i] = gamma * positive - gamma * y[i] * total
    
    return dydt
        
        


def solve_ivp_model(gamma, y0, t):

<<<<<<< HEAD
    solution = solve_ivp(ivp_model, [t[0], t[-1]], y0, args=(gamma,), t_eval=t, method='Radau')    
   
    
    return solution.y.T
=======
    
    solution = solve_ivp(ivp_model, [t[0], t[-1]], y0, args=(gamma,), t_eval=t, method='Radau')
    
    
    return solution
>>>>>>> b8cda375a62a31263e566330b6724178c3afc139




