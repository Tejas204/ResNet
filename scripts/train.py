# Imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from datetime import datetime

# ------------------------------------------------------------------------------------------------------
# Define device
# ------------------------------------------------------------------------------------------------------
device = torch.device("mps" if torch.mps.is_available() else "cpu")


# ------------------------------------------------------------------------------------------------------
# Setup training class, config and loop
# ------------------------------------------------------------------------------------------------------
class TrainModel():
    def __init__(self, criterion, optimizer, num_epochs, train_loader, val_loader, model, learning_rate):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def train_evaluate(self):
        # ------------------------------------------------------------------------------------------------------
        # Model training
        # ------------------------------------------------------------------------------------------------------
        train_loss = 0.0
        self.model.train()
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Make prediction
                outputs = self.model(images)

                # Loss
                loss = self.criterion(outputs, labels)

                # Update train loss
                self.train_loss

                # Zero gradients
                self.optimizer.zero_grad()

                # Compute gradients
                loss.backward()

                # Update weights
                self.optimizer.step()

            # ------------------------------------------------------------------------------------------------------
            # Model evaluation: This is early stopping as a regularization method
            # ------------------------------------------------------------------------------------------------------
            # Set the model in evaluation mode
            self.model.eval()
            valid_loss = 0.0
            best_valid_loss = 1_000_000

            with torch.no_grad():
                for images, labels in self.val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    # Make prediction
                    outputs = self.model(images)

                    # Compute loss
                    loss = self.criterion(outputs, labels)

                    # Update validation loss
                    valid_loss += loss.item()

                print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(self.train_loader)} \t\t Validation Loss: {valid_loss / len(self.val_loader)}')

                # This is specialization to early stopping for 1 step lookback
                if best_valid_loss < valid_loss:
                    best_valid_loss = valid_loss
                    model_path = f'model_{self.timestamp}_{epoch}'
                    torch.save(self.model.state_dict(), model_path)

    
