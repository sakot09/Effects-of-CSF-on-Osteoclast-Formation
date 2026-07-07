import numpy as np
from scipy.optimize import minimize
from osteoclast_model import solve_ode_model
import matplotlib.pyplot as plt




present = np.loadtxt("csfdata.csv")

absent = np.loadtxt("csfabsent.csv")


x = [0,4,8,16,24,48,72]

ypres = present[:, 1]

yabs = absent[:, 1]


def ssr(params, t, y):
    gamma, M = params

    M_pred, O_pred = solve_ode_model(gamma, M, 0, t)

    resid = y - O_pred

    return np.sum(resid**2)

initial = [0.01, 9.824280504908833791e+02]

result_w_csf = minimize(ssr, initial, args=(x, ypres))
csf_gamma, csf_M = result_w_csf.x

print(f"WITH CSF: GAMMA = {csf_gamma} M = {csf_M}")

result_wo_csf = minimize(ssr, initial, args=(x, yabs))
no_gamma, no_M = result_wo_csf.x

print(f"WITHOUT CSF: GAMMA = {no_gamma} M = {no_M}")


x_smooth = np.linspace(0, 72, 1000)

MC, y_pred_csf = solve_ode_model(csf_gamma, csf_M, 9.824280504908833791e+02, x_smooth)

MA, y_pred_no_csf = solve_ode_model(no_gamma, no_M, 9.824280504908833791e+02, x_smooth)


plt.scatter(x, ypres, label="Data", color="black", s=20)
plt.plot(x_smooth, y_pred_csf, label="Best fit", color="red")
plt.xlabel("Time (h)")
plt.ylabel("Total # Osteoclasts")
plt.title("Osteoclast Formation in the Presence of CSF")
plt.legend()
plt.savefig("formation_with_csf.png")
plt.close()


plt.scatter(x, yabs, label="Data", color="black", s=20)
plt.plot(x_smooth, y_pred_no_csf, label="Best fit", color="red")
plt.xlabel("Time (h)")
plt.ylabel("Total # Osteoclasts")
plt.title("Osteoclast Formation in the Absence of CSF")
plt.legend()
plt.savefig("formation_without_csf.png")
plt.close()



