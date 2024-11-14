import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import os

# Define the directory where you want to save the image
output_dir = "output/"
os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist

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
        # y is the state vector, which has a size of [batch_size, input_dim]
        return self.net(y)  # Return the ODE function

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

# Load the data from the txt file with the correct data type
data = np.loadtxt('data_train.txt', delimiter=',', dtype=np.float32)

# Example Usage
input_dim = 6  # After dimensionality reduction, the input dimension becomes 6
hidden_dim = 50
output_dim = 6  # 6 output features

# Create Neural ODE Model
model = ODEModel(input_dim, hidden_dim, output_dim)

# Time range from 0 to 1
t_span = torch.linspace(0., 1.01, 102)  # Time steps

window_size = 100

# Reshape the inputs to (num_samples, 100, 6)
# Assuming each row is a time step with 7 columns: 1 for time and 6 for the states

time_steps = len(data)  # The total number of time steps (rows in data)
state_size = 6          # Number of state variables per time step (columns 2 to 7)

# Initialize the arrays for input and output data
input_data = np.zeros((time_steps - window_size, window_size, state_size))  # (num_samples, 100, 6)
output_data = np.zeros((time_steps - window_size, state_size))      # (num_samples, 6)

# Loop to create time-series inputs and outputs
for i in range(time_steps - window_size):
    input_data[i] = data[i:i + window_size, 1:7]  # Inputs: columns 2 to 7 for 100 time steps
    output_data[i] = data[i + window_size, 1:7]  # Output: columns 2 to 7 for the state at time t+100

# Convert to torch tensors
inputs_tensor = torch.tensor(input_data, dtype=torch.float32)
outputs_tensor = torch.tensor(output_data, dtype=torch.float32)

# Training loop
epochs = 500

criterion = nn.MSELoss()  # Mean squared error loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass: Get the model output
    output = model(inputs_tensor, t_span)

    # Compute loss: Compare output to target
    loss = criterion(output, outputs_tensor)

    # Backward pass: Compute gradients
    loss.backward()

    # Update the model parameters
    optimizer.step()

    # Print loss every 100 epochs
    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item()}")


# Convert predictions and targets to numpy for easier plotting
predicted_values = output.detach().numpy()  # Model predictions
true_values = output_data  # True target values from the dataset

# Plot each output feature over the sample indices
num_features = predicted_values.shape[1]
sample_indices = range(len(predicted_values))  # X-axis

# Save the plot to a file in the specified directory
output_path = os.path.join(output_dir, "train_result.png")

plt.figure(figsize=(15, 15))

for i in range(num_features):
    plt.subplot(2, 3, i + 1)  # Create a subplot for each feature
    plt.plot(sample_indices, true_values[:, i], label="True", color="blue")
    plt.plot(predicted_values[:, 1], label="Predicted", color="orange", linestyle="--")
    plt.xlabel("Sample index")
    plt.ylabel(f"Feature {i + 1}")
    plt.legend(loc="best")  # Position legend automatically to avoid overlap
    plt.title(f"Output Feature {i + 1}")

plt.savefig(output_path)  # Save the plot
plt.close()  # Close the plot to free up memory

# Load the data from the txt file with the correct data type
data = np.loadtxt('data_test.txt', delimiter=',', dtype=np.float32)

window_size = 100

time_steps = len(data)  # The total number of time steps (rows in data)
state_size = 6          # Number of state variables per time step (columns 2 to 7)

# Initialize the arrays for input and output data
input_data = np.zeros((time_steps - window_size, window_size, state_size))  # (num_samples, 100, 6)
output_data = np.zeros((time_steps - window_size, state_size))      # (num_samples, 6)

# Loop to create time-series inputs and outputs
for i in range(time_steps - window_size):
    input_data[i] = data[i:i + window_size, 1:7]  # Inputs: columns 2 to 7 for 100 time steps
    output_data[i] = data[i + window_size, 1:7]  # Output: columns 2 to 7 for the state at time t+100

# Convert to torch tensors
inputs_tensor = torch.tensor(input_data, dtype=torch.float32)
outputs_tensor = torch.tensor(output_data, dtype=torch.float32)

# Convert predictions and targets to numpy for easier plotting
output = model(inputs_tensor, t_span)
predicted_values = output.detach().numpy()  # Model predictions
true_values = output_data  # True target values from the dataset

# Plot each output feature over the sample indices
num_features = predicted_values.shape[1]
sample_indices = range(len(predicted_values))  # X-axis

# Save the plot to a file in the specified directory
output_path = os.path.join(output_dir, "test_result.png")

plt.figure(figsize=(15, 15))

for i in range(num_features):
    plt.subplot(2, 3, i + 1)  # Create a subplot for each feature
    plt.plot(sample_indices, true_values[:, i], label="True", color="blue")
    plt.plot(sample_indices, predicted_values[:, i], label="Predicted", color="orange")
    plt.xlabel("Sample index")
    plt.ylabel(f"Feature {i + 1}")
    plt.legend(loc="best")  # Position legend automatically to avoid overlap
    plt.title(f"Output Feature {i + 1}")

plt.savefig(output_path)  # Save the plot
plt.close()  # Close the plot to free up memory
