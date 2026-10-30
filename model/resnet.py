"""
-----------------------------------------------------------------------------------------------------------------------
ResNet Theory

1. ResNet makes use of residual connections.
    - With residual connections, one can construct deeper networks
    - It helps mitigate the problem of vanishing gradient and lower layers not receiving gradient updates
    - Sometimes, the network only needs to learn the identity function

2. Architecture

    - ResNet uses weight layer interleaved with RelU and skip connection

        x --> Weight layer --> ReLu --> Weight layer --> H(x)
        |                                                ^
        |                                                |
        ----------------------- x ------------------------

3. Comparison with CNN

-----------------------------------------------------------------------------------------------------------------------

"""

# Imports
from pathlib import Path

import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# ------------------------------------------------------------------------------------------------------
# Hyperparams: import from config file
# ------------------------------------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[1]
config_path = project_root / "config" / "configuration.yaml"

with config_path.open('r') as config_file:
    config = yaml.safe_load(config_file)

batch_size = config['model']['hyperparameters']['batch_size']


# ------------------------------------------------------------------------------------------------------
# Define device
# ------------------------------------------------------------------------------------------------------
device = torch.device("mps" if torch.mps.is_available() else "cpu")

# ------------------------------------------------------------------------------------------------------
# Setup the model
# ------------------------------------------------------------------------------------------------------
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_blocks, n_features):
        super(ResNet, self).__init__()
        """
        ------------------------------------------------------------------------------------------------------

        Args:
            in_channels
            out_channels
            n_blocks
            n_features
        
        Output:
            none

        The function defines the following
        - Convolution layers including 1x1 convolution layers.
        - Activation function: relu
        - batchnorm
        - linear layer

        ------------------------------------------------------------------------------------------------------
        """
        # Define colvolution layers
        self.conv_init = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=1, device=device)

        self.conv64 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, device=device)

        self.conv128_init = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, device=device)
        self.conv128 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, device=device)
        self.conv1x1_128 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, device=device)

        self.conv256_init = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, device=device)
        self.conv256 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, device=device)
        self.conv1x1_256 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, device=device)

        # Define pooling
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

        # Define batch norms
        self.batchnorm64 = nn.BatchNorm2d(num_features=64, device=device)
        self.batchnorm128 = nn.BatchNorm2d(num_features=128, device=device)
        self.batchnorm256 = nn.BatchNorm2d(num_features=256, device=device)
        
        # Define Relu
        self.relu = nn.ReLU()

        # Define flatten
        self.flatten = nn.Flatten()

        # Define linear layer
        self.fully_connected = nn.Linear(in_features=256, out_features=10, bias=True, device=device)


    def forward(self, x):
        """
        ------------------------------------------------------------------------------------------------------
        Args:
            x: input

        Output:
            logits of dimension 10

        - Call the resnet blocks "n" times as needed.
        - Within each block, BatchNorm is used before ReLu
        - There are 2 blocks each of 3x3 conv with 64, 128 and 256 filters followed by flattening and fc

        ------------------------------------------------------------------------------------------------------
        """

        # Initial 7x7 convolution and pool
        out = self.conv_init(x)
        out = self.batchnorm64(out)
        out = self.relu(out)
        out = self.pool(out)

        # 2 Blocks of 3x3 conv with 64 filters
        out64_1 = self.conv64(out)
        out64_1 = self.batchnorm64(out64_1)
        out64_1 = self.relu(out64_1)
        out64_1 = self.conv64(out64_1)
        out64_1 = self.batchnorm64(out64_1)
        out64_1 = self.relu(out64_1)
        out64_1 = out64_1 + out

        out64_2 = self.conv64(out64_1)
        out64_2 = self.batchnorm64(out64_2)
        out64_2 = self.relu(out64_2)
        out64_2 = self.conv64(out64_2)
        out64_2 = self.batchnorm64(out64_2)
        out64_2 = self.relu(out64_2)
        out64_2 = out64_2 + out64_1

        # 2 Blocks of 3x3 conv with 128 filters
        out128_1 = self.conv128_init(out64_2)
        out128_1 = self.batchnorm128(out128_1)
        out128_1 = self.relu(out128_1)
        out128_1 = self.conv128(out128_1)
        out128_1 = self.batchnorm128(out128_1)
        out128_1 = self.relu(out128_1)
        out128_1 = out128_1 + self.conv1x1_128(out64_2)

        out128_2 = self.conv128(out128_1)
        out128_2 = self.batchnorm128(out128_2)
        out128_2 = self.relu(out128_2)
        out128_2 = self.conv128(out128_2)
        out128_2 = self.batchnorm128(out128_2)
        out128_2 = self.relu(out128_2)
        out128_2 = out128_2 + out128_1

        # 2 Blocks of 3x3 conv with 256 filters
        out256_1 = self.conv256_init(out128_2)
        out256_1 = self.batchnorm256(out256_1)
        out256_1 = self.relu(out256_1)
        out256_1 = self.conv256(out256_1)
        out256_1 = self.batchnorm256(out256_1)
        out256_1 = self.relu(out256_1)
        out256_1 = out256_1 + self.conv1x1_256(out128_2)

        out256_2 = self.conv256(out256_1)
        out256_2 = self.batchnorm256(out256_2)
        out256_2 = self.relu(out256_2)
        out256_2 = self.conv256(out256_2)
        out256_2 = self.batchnorm256(out256_2)
        out256_2 = self.relu(out256_2)
        out256_2 = out256_2 + out256_1

        # Flatten by using global average pool
        out = torch.mean(out256_2, dim=[2, 3])

        # Linear projection
        out = self.fully_connected(out)

        return out
