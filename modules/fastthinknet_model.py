# Import necessary PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the neural network architecture
class FastThinkNet(nn.Module):
    def __init__(self):
        super(FastThinkNet, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(in_features=784, out_features=128)  # MNIST input
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # Output layer

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(-1, 784)
        # Apply layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Final layer
        return x

# Note: Do not instantiate the model here. This will be done in the main application code.