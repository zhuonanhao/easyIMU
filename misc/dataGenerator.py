import numpy as np
from scipy.integrate import odeint
import os

# Ensure the directory exists for saving data
os.makedirs("local_data/train", exist_ok=True)

def nonlinear_dynamics(x, t):
    # Parameters for the nonlinear damping and restoring effects
    alpha = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 1.0])  # Nonlinear terms
    beta = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.5])   # Linear damping terms
    
    dxdt = np.zeros_like(x)
    dxdt[0] = -alpha[0] * x[0]**3 - beta[0] * x[0]
    dxdt[1] = -alpha[1] * x[1]**2 * np.sin(x[1]) - beta[1] * x[1]
    dxdt[2] = -alpha[2] * x[2]**2 * x[0] - beta[2] * x[2]
    dxdt[3] = -alpha[3] * np.cos(x[3]) - beta[3] * x[3]
    dxdt[4] = -alpha[4] * x[4]**2 * np.sin(x[4]) - beta[4] * x[4]
    dxdt[5] = -alpha[5] * x[5]**3 - beta[5] * x[5]
    return dxdt

t = np.linspace(0.0, 10, 1001)   # Time points from 0 to 10 with 1000 intervals

# Generate and save the dataset with clamping
for i in range(20):
    # Set initial conditions and time points
    x0 = 10 * np.random.rand(6) - 5  # Initial state within the range [-10, 10]
    

    # Solve the system using odeint
    solution = odeint(nonlinear_dynamics, x0, t)

    # Save the data to 'dataset_i.txt'
    # First column is time, the next six columns are the states
    data = np.column_stack((t, solution))
    # filename = f"local_data/train/dataset_{i}.txt"
    filename = f"local_data/test/dataset_{i}.txt"
    np.savetxt(filename, data, fmt="%5.6f", delimiter=",")
    print(f"Simulation data saved to '{filename}'")
