import torch.nn as nn
import torch
import numpy as np
from torchdiffeq import odeint

def nonlinear_dynamics(x):
    # Parameters for the nonlinear damping and restoring effects as tensors
    alpha = torch.tensor([1.0, 1.0, 1.0, 0.5, 0.5, 1.0], device=x.device)  # Nonlinear terms
    beta = torch.tensor([0.5, 0.5, 0.5, 0.2, 0.2, 0.5], device=x.device)   # Linear damping terms
    
    # Initialize dxdt as a tensor with the same shape as x
    dxdt = torch.zeros_like(x)
    
    # Define the nonlinear dynamics for each element
    dxdt[0] = -alpha[0] * x[0]**3 - beta[0] * x[0]
    dxdt[1] = -alpha[1] * x[1]**2 * torch.sin(x[1]) - beta[1] * x[1]
    dxdt[2] = -alpha[2] * x[2]**2 * x[0] - beta[2] * x[2]
    dxdt[3] = -alpha[3] * torch.cos(x[3]) - beta[3] * x[3]
    dxdt[4] = -alpha[4] * x[4]**2 * torch.sin(x[4]) - beta[4] * x[4]
    dxdt[5] = -alpha[5] * x[5]**3 - beta[5] * x[5]
    
    return dxdt

# Define the ODE function using a neural network
class NeuralODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralODE, self).__init__()
        # Define the neural network (ODE function)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Output dimension set to 6
        )

    def forward(self, t, y):
        # Neural network output
        neural_output = self.net(y)

        # Physical model output
        physical_output = nonlinear_dynamics(y)

        # Combine the outputs with a 0.5 weight on each
        combined_output = 0.1 * neural_output + 0.9 * physical_output

        return combined_output  # Return the combined dynamics

# Define a model to reduce the input dimension
class DimensionalityReduction(nn.Module):
    def __init__(self):
        super(DimensionalityReduction, self).__init__()
        # Global average pooling to reduce input from 100 timesteps to 1 timestep
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x is of shape [batch_size, 100, 6]
        x = x.permute(0, 2, 1)  # Permute to [batch_size, 6, 100]
        x = self.pool(x)  # Apply adaptive pooling to reduce timesteps from 100 to 1
        return x.squeeze(-1)  # Squeeze to get shape [batch_size, 6]

# Example ODE Model
class ODEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ODEModel, self).__init__()
        self.dimensionality_reduction = DimensionalityReduction()
        self.neural_ode = NeuralODE(input_dim, hidden_dim, output_dim)

    def forward(self, x0, t_span):
        # Apply dimensionality reduction to the input
        x0 = self.dimensionality_reduction(x0)  # Now x0 is of shape [batch_size, 6]
        y = odeint(self.neural_ode, x0, t_span)

        # Solve ODE from t=0 to t=end using odeint
        return y[-1]