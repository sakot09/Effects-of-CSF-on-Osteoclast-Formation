#no curve fit use minimize
import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt



data = np.loadtxt("gaussian (1).dat")

x = data[:, 0]
y = data[:, 1]

def gaussian(A, m, s, x):
    return A*(np.exp((-(x-m)**2)/s))


def ssr(params, x, y):
    A, m, s = params

    y_pred = gaussian(A, m, s, x)

    residual = y - y_pred

    return np.sum(residual**2)

initial = [10, 1, 5]

result = minimize(ssr, initial, args=(x, y))

best_a, best_m, best_s = result.x


print(f"A = {best_a:.4f}")
print(f"m = {best_m:.4f}")
print(f"s = {best_s:.4f}")
print(f"SSR = {result.fun:.4f}")
 

x_smooth = np.linspace(-10, 10, 1000)
y_fit = gaussian( best_a, best_m, best_s, x_smooth)
 
plt.scatter(x, y, label="Data", color="black", s=20)
plt.plot(x_smooth, y_fit, label="Best fit", color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Gaussian Fit")
plt.legend()
plt.savefig("problem2_gaussian.png", dpi=150)
print("Plot saved.")
