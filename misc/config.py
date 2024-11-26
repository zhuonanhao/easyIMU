import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters for the nonlinear damping and restoring effects
alpha = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 1.0])  # Nonlinear terms
beta = np.array([0.5, 0.5, 0.5, 0.2, 0.2, 0.5])   # Linear damping terms

# Define the nonlinear 6-state dynamics
def nonlinear_dynamics(t, x):
    dxdt = np.zeros_like(x)
    dxdt[0] = -alpha[0] * x[0]**3 - beta[0] * x[0]
    dxdt[1] = -alpha[1] * x[1]**2 * np.sin(x[1]) - beta[1] * x[1]
    dxdt[2] = -alpha[2] * x[2]**2 * x[0] - beta[2] * x[2]
    dxdt[3] = -alpha[3] * np.cos(x[3]) - beta[3] * x[3]
    dxdt[4] = -alpha[4] * x[4]**2 * np.sin(x[4]) - beta[4] * x[4]
    dxdt[5] = -alpha[5] * x[5]**3 - beta[5] * x[5]
    return dxdt

# Initial conditions for the system (arbitrarily chosen to observe the stability behavior)
x0 = np.array([1.0, 0.5, -0.5, 1.5, -1.0, 0.8])

# Time span for the simulation
t_span = (0, 20)  # From t=0 to t=20
t_eval = np.linspace(t_span[0], t_span[1], 500)  # Points at which to evaluate the solution

# Solve the ODE
solution = solve_ivp(nonlinear_dynamics, t_span, x0, t_eval=t_eval)

# Plot the solution
plt.figure(figsize=(12, 8))
for i in range(6):
    plt.plot(solution.t, solution.y[i], label=f'State x{i+1}')
plt.xlabel('Time')
plt.ylabel('State values')
plt.legend()
plt.title('Stable Nonlinear Dynamics of a 6-State System')
plt.show()
