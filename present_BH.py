import numpy as np
from scipy.optimize import basinhopping
from model import solve_ivp_model
import matplotlib.pyplot as plt




osteo_present = np.loadtxt("F4wCSF.csv", delimiter= ",")

nuclei_present = np.loadtxt("F5wCSF.csv", delimiter= ",")

f6_present = np.loadtxt("F6WCSF.csv")

x = [0,4,8,16,24,48,72]

y_osteo_present = osteo_present[:, 1]
y_ost_pre_scaled = y_osteo_present/np.max(y_osteo_present)

y_nuclei_present = nuclei_present[:, 1]
y_nuc_pre_scaled = y_nuclei_present/np.max(y_nuclei_present)


y_f6_present = f6_present[:,1]
y_f6_pre_scaled = y_f6_present/np.max(y_f6_present)



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


initial = np.full(11, np.log10(1e-8))
initial[0] = np.log10(1e-4)
initial[1] = np.log10(186)


result_csf = basinhopping(ssr, initial, minimizer_kwargs= {"args": (x, y_ost_pre_scaled, y_nuc_pre_scaled, y_f6_pre_scaled)})


csf_gamma = 10**result_csf.x[0]
csf_y0 = 10**result_csf.x[1:]
csf_y0_full = np.full(50,1e-8)
csf_y0_full[:10] = csf_y0



print(f"WITH CSF: gamma = {csf_gamma}")

csf_preds = solve_ivp_model(csf_gamma, csf_y0_full, x)

total_csf = np.sum(csf_preds.y.T, axis=1)
mean_nuclei_csf = np.sum((np.arange(1, 51) * csf_preds.y.T), axis = 1) / total_csf


csf_compartments = csf_preds.y.T[3, :]
csf_bins = np.array([
    csf_compartments[1],
    csf_compartments[2],
    csf_compartments[3],
    csf_compartments[4],
    csf_compartments[5],
    csf_compartments[6] + csf_compartments[7],
    csf_compartments[8] + csf_compartments[9],
    np.sum(csf_compartments[10:14]),
    np.sum(csf_compartments[14:18]),
    np.sum(csf_compartments[18:24]),
    np.sum(csf_compartments[24:29]),
    np.sum(csf_compartments[29:39]),
    np.sum(csf_compartments[39:49]),
    csf_compartments[49]
])

with open("results_BH_present.txt", "w") as f:
    f.write(f"WITH CSF: gamma = {csf_gamma}\n")
    f.write(f"success: {result_csf.success}\n")

bin_labels = ['2', '3', '4', '5', '6', '7-8', '9-10', '11-14', '15-18', '19-24', '25-29', '30-39', '40-49', '>50']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))

ax1.plot(x, y_osteo_present, 'o', color='blue', label='CSF data')
ax1.plot(x, total_csf, color='blue', label='CSF model')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Total Osteoclasts')
ax1.set_title('Figure 4')
ax1.legend()

ax2.plot(x, y_nuclei_present, 'o', color='blue', label='CSF data')
ax2.plot(x, mean_nuclei_csf, color='blue', label='CSF model')
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Mean Nuclei per Osteoclast')
ax2.set_title('Figure 5')
ax2.legend()

ax3.plot(bin_labels, y_f6_present, 'o', color='blue', label='CSF data')
ax3.plot(bin_labels, csf_bins, color='blue', label='CSF model')
ax3.set_xlabel('Nuclei per Osteoclast')
ax3.set_ylabel('Number of Osteoclasts')
ax3.set_title('Figure 6')
ax3.tick_params(axis='x', rotation=45)
ax3.legend()

plt.tight_layout()
plt.savefig("plot_BH_present.png")
