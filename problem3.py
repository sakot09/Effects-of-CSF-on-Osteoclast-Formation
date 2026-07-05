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


y_fit = gaussian( best_a, best_m, best_s, x)

residuals = y-y_fit


a_list = []
m_list = []
s_list = []

for i in range(100):
    shuffled_residuals = np.random.permutation(residuals)

    new_y = y_fit+shuffled_residuals
    result = minimize(ssr, initial, args=(x, new_y))

    test_a, test_m, test_s = result.x
    a_list.append(test_a)
    m_list.append(test_m)
    s_list.append(test_s)

a_list.sort()
m_list.sort()
s_list.sort()

print("95% Confidence Intervals")

print(f"A: ({a_list[1]}, {a_list[-2]})")
print(f"M: ({m_list[1]}, {m_list[-2]})")
print(f"S: ({s_list[1]}, {s_list[-2]})")