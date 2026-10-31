# Imports
from pathlib import Path
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import resnet
from loaders import data_loader
from scripts import test
from scripts import train

def main():
    # Define model config
    config_path = PROJECT_ROOT / "config" / "configuration.yaml"
    with config_path.open('r') as config_file:
        config = yaml.safe_load(config_file)

    n_blocks = config['model']['hyperparameters']['n_blocks']
    n_features = config['model']['hyperparameters']['n_features']
    num_epochs = config['model']['hyperparameters']['num_epochs']
    learning_rate = config['model']['hyperparameters']['learning_rate']

    # Define dataloaders
    train_loader, val_loader, test_loader = data_loader.DataLoaders().load_data()

    # Define model
    model = resnet.ResNet(in_channels=1, out_channels=3, n_blocks=n_blocks, n_features=n_features)

    # Train model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trainer = train.TrainModel(criterion=criterion, optimizer=optimizer, num_epochs=num_epochs,
                               train_loader=train_loader, val_loader=val_loader, model=model, learning_rate=learning_rate)
    trainer.train_evaluate()

    # Test model
    tester = test.TestModel(model=model, loader=test_loader)
    tester.evaluate()


if __name__ == "__main__":
    main()
