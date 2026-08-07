import numpy as np
from scipy.optimize import dual_annealing
from model import solve_ivp_model
import matplotlib.pyplot as plt




osteo_absent = np.loadtxt("F4noCSF.csv", delimiter= ",")

nuclei_absent = np.loadtxt("F5noCSF.csv", delimiter= ",")

f6_absent = np.loadtxt("F6noCSF.csv")

x = [0,4,8,16,24,48,72]

y_osteo_absent = osteo_absent[:,1]
y_ost_abs_scaled = y_osteo_absent/np.max(y_osteo_absent)

y_nuclei_absent = nuclei_absent[:, 1]
y_nuc_abs_scaled = y_nuclei_absent/np.max(y_nuclei_absent)


y_f6_absent = f6_absent[:,1]
y_f6_abs_scaled = y_f6_absent/np.max(y_f6_absent)



def ssr(params, t, y_total, y_nuclei, y_f6):

    params = 10**params

    gamma = params[0]

    y0_full = np.full(50,1e-8)
    y0_full[:10] = params[1:]



    sol = solve_ivp_model(gamma, y0_full, t)

    solution = sol.y.T

    total = np.sum(solution, axis=1)
    mean_nuclei = np.sum((np.arange(1, 51) * solution), axis = 1) / total
    
    resid_osteo = (y_total - total)/y_total
    resid_nuclei = (y_nuclei - mean_nuclei)/y_nuclei

    compartments = solution[3,:]

    bins = [
        compartments[1],
        compartments[2],
        compartments[3],
        compartments[4],
        compartments[5],
        compartments[6] + compartments[7],
        compartments[8] + compartments[9],
        np.sum(compartments[10:14]),
        np.sum(compartments[14:18]),
        np.sum(compartments[18:24]),
        np.sum(compartments[24:29]),
        np.sum(compartments[29:39]),
        np.sum(compartments[39:49]),
        compartments[49]
    ]

    bins = np.array(bins)
    resid_f6 = (y_f6 - bins)/y_f6
    
    return np.sum(resid_osteo**2) + np.sum(resid_nuclei**2) + np.sum(resid_f6**2)


bounds = [(-10,0)]
bounds += [(-8,4)] * 10


result_absent = dual_annealing(ssr, bounds, args=(x, y_ost_abs_scaled, y_nuc_abs_scaled, y_f6_abs_scaled))




absent_gamma = 10**result_absent.x[0]
absent_y0 = 10**result_absent.x[1:]
absent_y0_full = np.full(50,1e-8)
absent_y0_full[:10] = absent_y0





absent_preds = solve_ivp_model(absent_gamma, absent_y0_full, x)


absent_compartments = absent_preds.y.T[3, :]
absent_bins = np.array([
    absent_compartments[1],
    absent_compartments[2],
    absent_compartments[3],
    absent_compartments[4],
    absent_compartments[5],
    absent_compartments[6] + absent_compartments[7],
    absent_compartments[8] + absent_compartments[9],
    np.sum(absent_compartments[10:14]),
    np.sum(absent_compartments[14:18]),
    np.sum(absent_compartments[18:24]),
    np.sum(absent_compartments[24:29]),
    np.sum(absent_compartments[29:39]),
    np.sum(absent_compartments[39:49]),
    absent_compartments[49]
])

bin_labels = ['2', '3', '4', '5', '6', '7-8', '9-10', '11-14', '15-18', '19-24', '25-29', '30-39', '40-49', '>50']

total_absent = np.sum(absent_preds.y.T, axis=1)
mean_nuclei_absent = np.sum((np.arange(1, 51) * absent_preds.y.T), axis = 1) / total_absent

with open("results_DA_absent.txt", "w") as f:
    f.write(f"WITHOUT CSF: gamma = {absent_gamma}\n")
    f.write(f"success: {result_absent.success}\n")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))


ax1.plot(x, y_osteo_absent, 'o', color='red', label='No CSF data')
ax1.plot(x, total_absent, color='red', label='No CSF model')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Total Osteoclasts')
ax1.set_title('Figure 4')
ax1.legend()

ax2.plot(x, y_nuclei_absent, 'o', color='red', label='No CSF data')
ax2.plot(x, mean_nuclei_absent, color='red', label='No CSF model')
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Mean Nuclei per Osteoclast')
ax2.set_title('Figure 5')
ax2.legend()

ax3.plot(bin_labels, y_f6_absent, 'o', color='red', label='No CSF data')
ax3.plot(bin_labels, absent_bins, color='red', label='No CSF model')
ax3.set_xlabel('Nuclei per Osteoclast')
ax3.set_ylabel('Number of Osteoclasts')
ax3.set_title('Figure 6')
ax3.tick_params(axis='x', rotation=45)
ax3.legend()

plt.tight_layout()
plt.savefig("plot_DA_absent.png")
