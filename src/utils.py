import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# Custom Dataset
class IMUDataDataset(Dataset):
    def __init__(self, data_dir, window_size, input_dim, output_dim):
        self.data_dir = data_dir
        self.window_size = window_size
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.all_inputs_tensor = []
        self.all_outputs_tensor = []

        data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.startswith('dataset') and file.endswith('.txt')]

        # Loop through all datasets to load data and concatenate them
        for data_file in data_files:
            print(f"Processing dataset: {data_file}")

            # Load input and output data, ignoring the time column
            time_data = np.loadtxt(data_file, delimiter=',', dtype=np.float32)[:, 0:1]
            input_data = np.loadtxt(data_file, delimiter=',', dtype=np.float32)[:, 1:7]  # Shape: (301, 6)
            output_data = np.loadtxt(data_file, delimiter=',', dtype=np.float32)[:, 7:13]  # Shape: (301, 6)

            # Prepare input and output data for training
            num_samples = len(input_data) - self.window_size
            input_tensor = np.zeros((num_samples, self.window_size, self.input_dim))  # Shape: (201, 100, 6)
            output_tensor = np.zeros((num_samples, self.output_dim))  # Shape: (201, 6)

            for i in range(num_samples):
                input_tensor[i] = input_data[i:i + self.window_size]
                output_tensor[i] = output_data[i + self.window_size:i + self.window_size + 1]

            # Append to the overall input and output tensors
            self.all_inputs_tensor.append(input_tensor)
            self.all_outputs_tensor.append(output_tensor)

        # Concatenate all input and output tensors
        self.inputs_tensor = np.concatenate(self.all_inputs_tensor, axis=0)  # Concatenate along the first axis
        self.outputs_tensor = np.concatenate(self.all_outputs_tensor, axis=0)

        # Convert to PyTorch tensors
        self.inputs_tensor = torch.tensor(self.inputs_tensor, dtype=torch.float32)
        self.outputs_tensor = torch.tensor(self.outputs_tensor, dtype=torch.float32)

        print(f"Input data size (batchsize x windowsize x numInputFeatures): {self.inputs_tensor.shape}")
        print(f"Output data size (batchsize x 1 x numOutputFeatures): {self.outputs_tensor.shape}")

    def __len__(self):
        # Return the number of samples
        return len(self.inputs_tensor)

    def __getitem__(self, idx):
        # Return the input and output tensors for a given index
        return self.inputs_tensor[idx], self.outputs_tensor[idx]
