import numpy as np
from scipy.optimize import minimize
from model import solve_ivp_model
import matplotlib.pyplot as plt




osteo_absent = np.loadtxt("F4noCSF.csv", delimiter= ",")
osteo_present = np.loadtxt("F4wCSF.csv", delimiter= ",")

nuclei_absent = np.loadtxt("F5noCSF.csv", delimiter= ",")
nuclei_present = np.loadtxt("F5wCSF.csv", delimiter= ",")


x = [0,4,8,16,24,48,72]

y_osteo_absent = osteo_absent[:,1]
y_osteo_present = osteo_present[:, 1]

y_nuclei_absent = nuclei_absent[:, 1]
y_nuclei_present = nuclei_present[:, 1]



def ssr(params, t, y_total, y_nuclei):
<<<<<<< HEAD
    params = 10**params
    
=======

    params = 10**params

>>>>>>> b8cda375a62a31263e566330b6724178c3afc139
    gamma = params[0]

    y0 = params[1:]

<<<<<<< HEAD
    solution = solve_ivp_model(gamma, y0, t)
=======
    sol = solve_ivp_model(gamma, y0, t)

    solution = sol.y.T
>>>>>>> b8cda375a62a31263e566330b6724178c3afc139

    total = np.sum(solution, axis=1)
    mean_nuclei = np.sum((np.arange(1, 51) * solution), axis = 1) / total
    
    resid_osteo = y_total - total
    resid_nuclei = y_nuclei - mean_nuclei

    print(f"gamma = {gamma:.2e}, SSR = {np.sum(resid_osteo**2) + np.sum(resid_nuclei**2):.4f}")
    
    return np.sum(resid_osteo**2) + np.sum(resid_nuclei**2)

initial_y0 = np.full(50, 1e-8)
<<<<<<< HEAD
initial_y0[0] = 186
initial = np.log10(np.concatenate(([1e-4], initial_y0)))

result_csf = minimize(ssr, initial, args=(x, y_osteo_present, y_nuclei_present), method='Nelder-Mead')

result_absent = minimize(ssr, initial, args=(x, y_osteo_absent, y_nuclei_absent))
=======
initial_y0[0] = 186  
initial = np.log10(np.concatenate(([1e-4], initial_y0)))

result_csf = minimize(ssr, initial, args=(x, y_osteo_present, y_nuclei_present), method='Nelder-Mead')
result_absent = minimize(ssr, initial, args=(x, y_osteo_absent, y_nuclei_absent), method='Nelder-Mead')
>>>>>>> b8cda375a62a31263e566330b6724178c3afc139

csf_gamma = result_csf.x[0]
csf_y0 = result_csf.x[1:]

absent_gamma = result_absent.x[0]
absent_y0 = result_absent.x[1:]

print(f"WITH CSF: gamma = {10**csf_gamma}")
print(f"WITHOUT CSF: gamma = {10**absent_gamma}")

