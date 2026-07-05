import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
 
 
def viral_model(y, t, beta, p, delta, c):
    
    T, I, V = y
 
    dTdt = -beta * T * V
    dIdt = beta * T * V - delta * I
    dVdt = p * I - c * V
 
    return [dTdt, dIdt, dVdt]
 
 
def solve_viral_model(beta, p, delta, c, T0, I0, V0, t):
    
    y0 = [T0, I0, V0]
    solution = odeint(viral_model, y0, t, args=(beta, p, delta, c))
 
    T = solution[:, 0]
    I = solution[:, 1]
    V = solution[:, 2]
 
    return T, I, V
 
 
if __name__ == "__main__":
    
    beta = 1e-5   
    p = 2e6       
    delta = 4     
    c = 4   

    T0 = 1
    I0 = 0
    V0 = 0.01
 
    
    t = np.linspace(0, 200, 1000)  
 
    T, I, V = solve_viral_model(beta, p, delta, c, T0, I0, V0, t)
 
   
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    axes[0].plot(t, T, label="Target")
    axes[0].plot(t, I, label="Infected")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Cell population")
    axes[0].legend()
    axes[0].set_title("Target and Infected Cells")
 
    axes[1].plot(t, V, color="red")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Virus titer (log scale)")
    axes[1].set_title("Load")
 
    plt.tight_layout()
    plt.savefig("problem1_viral_model.png", dpi=150)
    print("Plot saved.")      