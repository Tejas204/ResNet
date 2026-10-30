# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# ------------------------------------------------------------------------------------------------------
# Define device
# ------------------------------------------------------------------------------------------------------
device = torch.device("mps" if torch.mps.is_available() else "cpu")


# ------------------------------------------------------------------------------------------------------
# Setup training class, config and loop
# ------------------------------------------------------------------------------------------------------
class TrainModel():
    def __init__(self, criterion, optimizer, num_epochs, loader, model, learning_rate):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.loader = loader
        self.learning_rate = learning_rate

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.loader):
                images = images.to(device)
                labels = labels.to(device)

                # Make prediction
                outputs = self.model(images)

                # Loss
                loss = criterion(outputs, labels)

                # Zero gradients
                optimizer.zero_grad()

                # Compute gradients
                loss.backward()

                # Update weights
                optimizer.step()


                if (i+1)%100 == 0:
                    print(f"Epoch: {epoch}/{self.num_epochs}, loss: {loss.item()}")