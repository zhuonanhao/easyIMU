import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import os

from model import ODEModel

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define directories for checkpoints and loss plot
checkpoint_dir = "output/checkpoints/"
plot_dir = "output/plots/"
os.makedirs(checkpoint_dir, exist_ok=True)  # Create checkpoints folder if it doesn't exist
os.makedirs(plot_dir, exist_ok=True)        # Create plots folder if it doesn't exist

# Define the directory where the data files are located
data_dir = 'local_data/train'

# Example Usage
input_dim = 6  # After dimensionality reduction, the input dimension becomes 6
hidden_dim = 50
output_dim = 6  # 6 output features

# Create Neural ODE Model and move to GPU
model = ODEModel(input_dim, hidden_dim, output_dim).to(device)

# Time range from 0 to 1
t_span = torch.linspace(0, 1.01, 102).to(device)  # Time steps on GPU

window_size = 100

# Initialize empty lists to hold the final data
all_inputs = []
all_outputs = []

# Load and process data files
for file_name in os.listdir(data_dir):
    if file_name.endswith('.txt'):
        file_path = os.path.join(data_dir, file_name)
        data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        time_steps = len(data)  # Total number of time steps
        state_size = 6          # Number of state variables

        # Prepare input and output data for training
        input_data = np.zeros((time_steps - window_size, window_size, state_size))
        output_data = np.zeros((time_steps - window_size, state_size))

        for i in range(time_steps - window_size):
            input_data[i] = data[i:i + window_size, 1:7]
            output_data[i] = data[i + window_size, 1:7]

        all_inputs.append(input_data)
        all_outputs.append(output_data)

# Convert data lists to tensors and move to GPU
all_inputs = np.vstack(all_inputs)
all_outputs = np.vstack(all_outputs)
inputs_tensor = torch.tensor(all_inputs, dtype=torch.float32).to(device)
outputs_tensor = torch.tensor(all_outputs, dtype=torch.float32).to(device)

# Training setup
epochs = 500
checkpoint_interval = 50
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# List to store loss for plotting
loss_values = []

# Training loop
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    output = model(inputs_tensor, t_span)
    loss = criterion(output, outputs_tensor)
    loss_values.append(loss.item())  # Save loss for plotting

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

    # Save model checkpoint every 50 epochs
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

# Save the final model checkpoint
checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epochs}.pth")
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}, checkpoint_path)
print(f"The final checkpoint saved at epoch {epochs}")

# Plot the loss and save it
plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
loss_plot_path = os.path.join(plot_dir, 'training_loss.png')
plt.savefig(loss_plot_path)
print(f"Training loss plot saved to '{loss_plot_path}'")
