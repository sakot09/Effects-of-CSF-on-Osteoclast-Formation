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

    params = 10**params

    gamma = params[0]

    y0 = params[1:]

    sol = solve_ivp_model(gamma, y0, t)

    solution = sol.y.T

    total = np.sum(solution, axis=1)
    mean_nuclei = np.sum((np.arange(1, 51) * solution), axis = 1) / total
    
    resid_osteo = y_total - total
    resid_nuclei = y_nuclei - mean_nuclei

    print(f"gamma = {gamma:.2e}, SSR = {np.sum(resid_osteo**2) + np.sum(resid_nuclei**2):.4f}")
    
    return np.sum(resid_osteo**2) + np.sum(resid_nuclei**2)

initial_y0 = np.full(50, 1e-8)

initial_y0[0] = 186
initial = np.log10(np.concatenate(([1e-4], initial_y0)))

result_csf = minimize(ssr, initial, args=(x, y_osteo_present, y_nuclei_present), method='Nelder-Mead', options={"maxiter": 1000}, tol=1e-3)
result_absent = minimize(ssr, initial, args=(x, y_osteo_absent, y_nuclei_absent), method='Nelder-Mead', options={"maxiter": 1000}, tol=1e-3)


csf_gamma = 10**result_csf.x[0]
csf_y0 = 10**result_csf.x[1:]

absent_gamma = 10**result_absent.x[0]
absent_y0 = 10**result_absent.x[1:]

print(f"WITH CSF: gamma = {csf_gamma}")
print(f"WITHOUT CSF: gamma = {absent_gamma}")

print(result_csf.success)
print(result_absent.success)

csf_preds = solve_ivp_model(csf_gamma, csf_y0, x)

total_csf = np.sum(csf_preds.y.T, axis=1)
mean_nuclei_csf = np.sum((np.arange(1, 51) * csf_preds.y.T), axis = 1) / total_csf

absent_preds = solve_ivp_model(absent_gamma, absent_y0, x)

total_absent = np.sum(absent_preds.y.T, axis=1)
mean_nuclei_absent = np.sum((np.arange(1, 51) * absent_preds.y.T), axis = 1) / total_absent

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(x, y_osteo_present, 'o', color='blue', label='CSF data')
ax1.plot(x, total_csf, color='blue', label='CSF model')
ax1.plot(x, y_osteo_absent, 'o', color='red', label='No CSF data')
ax1.plot(x, total_absent, color='red', label='No CSF model')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Total Osteoclasts')
ax1.set_title('Figure 4')
ax1.legend()

ax2.plot(x, y_nuclei_present, 'o', color='blue', label='CSF data')
ax2.plot(x, mean_nuclei_csf, color='blue', label='CSF model')
ax2.plot(x, y_nuclei_absent, 'o', color='red', label='No CSF data')
ax2.plot(x, mean_nuclei_absent, color='red', label='No CSF model')
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Mean Nuclei per Osteoclast')
ax2.set_title('Figure 5')
ax2.legend()

plt.tight_layout()
plt.show()