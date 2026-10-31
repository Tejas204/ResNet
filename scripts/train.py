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
        self.model.train()
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.loader):
                images = images.to(device)
                labels = labels.to(device)

                # Make prediction
                outputs = self.model(images)

                # Loss
                loss = self.criterion(outputs, labels)

                # Zero gradients
                self.optimizer.zero_grad()

                # Compute gradients
                loss.backward()

                # Update weights
                self.optimizer.step()


                if (i+1)%100 == 0:
                    print(f"Epoch: {epoch+1}/{self.num_epochs}, loss: {loss.item()}")