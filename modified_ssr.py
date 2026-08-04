import numpy as np
from scipy.optimize import dual_annealing
from model import solve_ivp_model
import matplotlib.pyplot as plt




osteo_absent = np.loadtxt("F4noCSF.csv", delimiter= ",")
osteo_present = np.loadtxt("F4wCSF.csv", delimiter= ",")

nuclei_absent = np.loadtxt("F5noCSF.csv", delimiter= ",")
nuclei_present = np.loadtxt("F5wCSF.csv", delimiter= ",")

f6_absent = np.loadtxt("F6noCSF.csv")
f6_present = np.loadtxt("F6WCSF.csv")

x = [0,4,8,16,24,48,72]

y_osteo_absent = osteo_absent[:,1]
y_osteo_present = osteo_present[:, 1]

y_nuclei_absent = nuclei_absent[:, 1]
y_nuclei_present = nuclei_present[:, 1]

y_f6_absent = f6_absent[:,1]
y_f6_present = f6_present[:,1]



def ssr(params, t, y_total, y_nuclei, y_f6):

    params = 10**params

    gamma = params[0]

    y0 = params[1:]

    sol = solve_ivp_model(gamma, y0, t)

    solution = sol.y.T

    total = np.sum(solution, axis=1)
    mean_nuclei = np.sum((np.arange(1, 51) * solution), axis = 1) / total
    
    resid_osteo = y_total - total
    resid_nuclei = y_nuclei - mean_nuclei

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
    resid_f6 = y_f6 - bins
    
    return np.sum(resid_osteo**2) + np.sum(resid_nuclei**2) + np.sum(resid_f6**2)

def calculate_results():

    bounds = [(-10,0)]
    bounds += [(-8,4)] * 50
    

    result_csf = dual_annealing(ssr, bounds, args=(x, y_osteo_present, y_nuclei_present, y_f6_present))
    result_absent = dual_annealing(ssr, bounds, args=(x, y_osteo_absent, y_nuclei_absent, y_f6_absent))


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

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))

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

    ax3.plot(bin_labels, y_f6_present, 'o', color='blue', label='CSF data')
    ax3.plot(bin_labels, csf_bins, color='blue', label='CSF model')
    ax3.plot(bin_labels, y_f6_absent, 'o', color='red', label='No CSF data')
    ax3.plot(bin_labels, absent_bins, color='red', label='No CSF model')
    ax3.set_xlabel('Nuclei per Osteoclast')
    ax3.set_ylabel('Number of Osteoclasts')
    ax3.set_title('Figure 6')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend()

    plt.tight_layout()
    plt.show()

calculate_results()