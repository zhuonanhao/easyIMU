import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import os
from model import ODEModel
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from utils import IMUDataDataset  # Assuming you have defined this dataset class

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
epochs = 50
checkpoint_interval = 50
learning_rate = 0.01
batch_size = 32  # Define batch size for DataLoader

# Create the custom dataset
dataset = IMUDataDataset(data_dir, window_size, input_dim, output_dim)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


# Create Neural ODE Model and move to GPU
model = ODEModel(input_dim, hidden_dim, output_dim).to(device)

# Time range for the ODE solver
dt = 0.01  # Placeholder for actual time step from the data
t_span = torch.linspace(0, window_size * dt, window_size + 1).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
# Alternatively, use ReduceLROnPlateau:
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# List to store loss for plotting
loss_values = []

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    # Iterate through the dataloader
    for inputs, outputs in dataloader:
        inputs, outputs = inputs.to(device), outputs.to(device)  # Move to GPU
        
        optimizer.zero_grad()

        # Forward pass
        output = model(inputs, t_span)
        loss = criterion(output, outputs)
        running_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Calculate average loss for the epoch
    avg_loss = running_loss / len(dataloader)
    loss_values.append(avg_loss)  # Save loss for plotting

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    # Update learning rate using the scheduler
    scheduler.step()

    # Save model checkpoint every 50 epochs
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

# Save the final model checkpoint
checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epochs}.pth")
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
}, checkpoint_path)
print(f"Final checkpoint saved at epoch {epochs}")

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
