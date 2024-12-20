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

# Parameters
window_size = 1
input_dim = 6  # Number of input features
hidden_dim = 50
output_dim = 6  # Number of output features
epochs = 500
checkpoint_interval = 50
learning_rate = 0.01

# Get all dataset files in the data folder
data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.startswith('dataset') and file.endswith('.txt')]

# Lists to hold the input and output tensors
all_inputs_tensor = []
all_outputs_tensor = []

# Loop through all datasets to load data and concatenate them
for data_file in data_files:
    print(f"Processing dataset: {data_file}")

    # Load input and output data, ignoring the time column
    time_data = np.loadtxt(data_file, delimiter=',', dtype=np.float32)[:, 0:1]  
    input_data = np.loadtxt(data_file, delimiter=',', dtype=np.float32)[:, 1:7]  # Shape: (301, 6)
    output_data = np.loadtxt(data_file, delimiter=',', dtype=np.float32)[:, 7:13]  # Shape: (301, 6)

    # Prepare input and output data for training
    num_samples = len(input_data) - window_size
    input_tensor = np.zeros((num_samples, window_size, input_dim))  # Shape: (201, 100, 6)
    output_tensor = np.zeros((num_samples, output_dim))             # Shape: (201, 6)

    for i in range(num_samples):
        input_tensor[i] = input_data[i:i + window_size]
        output_tensor[i] = output_data[i + window_size:i + window_size + 1]

    # Append to the overall input and output tensors
    all_inputs_tensor.append(input_tensor)
    all_outputs_tensor.append(output_tensor)

dt = float(time_data[1] - time_data[0])

# Concatenate all input and output tensors
inputs_tensor = np.concatenate(all_inputs_tensor, axis=0)  # Concatenate along the first axis
outputs_tensor = np.concatenate(all_outputs_tensor, axis=0)

# Convert to PyTorch tensors and move to GPU
inputs_tensor = torch.tensor(inputs_tensor, dtype=torch.float32).to(device)
outputs_tensor = torch.tensor(outputs_tensor, dtype=torch.float32).to(device)

print(f"Input data size (batchsize x windowsize x numInputFeatures): {inputs_tensor.shape}")
print(f"Output data size (batchsize x 1 x numOutputFeatures): {outputs_tensor.shape}")

# Create Neural ODE Model and move to GPU
model = ODEModel(input_dim, hidden_dim, output_dim).to(device)

# Time range for the ODE solver
t_span = torch.linspace(0, window_size * dt, window_size + 1).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
