from utils import IMUDataDataset
from torch.utils.data import Dataset, DataLoader

# Parameters
data_dir = 'local_data/train'
window_size = 1
input_dim = 6  # Number of input features
output_dim = 6  # Number of output features

# Create the custom dataset
dataset = IMUDataDataset(data_dir, window_size, input_dim, output_dim)

# Create a DataLoader
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example: Iterate through the DataLoader
for inputs, outputs in dataloader:
    print(f"First input sample:\n{inputs[0]}")
    print(f"First output sample:\n{outputs[0]}")
    break  # Just to print the first batch as an example
