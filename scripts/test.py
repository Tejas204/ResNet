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
# Setup testing clsss, config and loop
# ------------------------------------------------------------------------------------------------------
class TestModel():
    def __init__(self, model, loader):
        self.model = model
        self.loader = loader

    def evaluate(self):
        n_correct = 0
        n_samples = 0
        for i, (images, labels) in enumerate(self.loader):
            images = images.to(device)
            labels = labels.to(device)
    
            # Make prediction
            outputs = self.model(images)
    
            _, predictions = torch.max(outputs, 1)
                    
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()
    
        acc = 100.0 * n_correct / n_samples
        print(f"Accuracy on test set: {acc}")