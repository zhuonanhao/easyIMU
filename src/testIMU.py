import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import ODEModel

# Define the directory where you want to save the image
result_dir = "output/result"
os.makedirs(result_dir, exist_ok=True)  # Create the folder if it doesn't exist

# Load pretrained model
checkpoint_path = 'output/checkpoints/model_epoch_500.pth'
checkpoint = torch.load(checkpoint_path)

# Recreate model architecture
input_dim = 6
hidden_dim = 50
output_dim = 6
model = ODEModel(input_dim, hidden_dim, output_dim)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Time range from 0 to 1
t_span = torch.linspace(0., 1.01, 102)  # Time steps

window_size = 100

# Iterate over all test files in the 'local_data' folder
for file_name in os.listdir('local_data/test/'):
    # Only process files matching the pattern 'test_X.txt'
    if file_name.endswith('.txt'):
        data_path = os.path.join('local_data/train/', file_name)

        # Load the data from the txt file with the correct data type
        data = np.loadtxt(data_path, delimiter=',', dtype=np.float32)

        time_steps = len(data)  # The total number of time steps (rows in data)
        state_size = 6          # Number of state variables per time step (columns 2 to 7)

        # Initialize the arrays for input and output data
        input_data = np.zeros((time_steps - window_size, window_size, state_size))  # (num_samples, 100, 6)
        output_data = np.zeros((time_steps - window_size, state_size))              # (num_samples, 6)

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

        # Get the test number from the file name (e.g., test_0.txt -> 0)
        test_number = file_name.split('.')[0].split('_')[1]

        # Save the plot to a file in the specified directory with the same name as the test file
        output_path = os.path.join(result_dir, f"test_{test_number}.png")

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

        print(f"Results saved for {file_name} as test_{test_number}.png")
