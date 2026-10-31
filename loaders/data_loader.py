# Imports
from pathlib import Path
import yaml
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

class DataLoaders():
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[1]
        config_path = self.project_root / "config" / "configuration.yaml"

        with config_path.open('r') as config_file:
            config = yaml.safe_load(config_file)

        self.batch_size = config['model']['hyperparameters']['batch_size']

    def load_data(self):
        # ------------------------------------------------------------------------------------------------------
        # Setup the data
        # ------------------------------------------------------------------------------------------------------
        data_path = self.project_root / "data"
        full_data = datasets.MNIST(root=data_path, train=True, transform=transforms.ToTensor(), download=True)
        train_data, validation_data = random_split(full_data, [50000, 10000])
        test_data = datasets.MNIST(root=data_path, train=False, transform=transforms.ToTensor(), download=True)
        

        # ------------------------------------------------------------------------------------------------------
        # Setup the data loader
        # ------------------------------------------------------------------------------------------------------
        train_loader = DataLoader(dataset=train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=validation_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
