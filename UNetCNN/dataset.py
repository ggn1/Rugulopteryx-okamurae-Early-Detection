# IMPORTS
import torch
from torch.utils.data import Dataset, DataLoader

class InfestationPairsDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # float32 1xHxW
        y = self.Y[idx]
        # Optionally add small noise / augmentation
        if self.transform:
            x, y = self.transform(x, y)
        return torch.from_numpy(x), torch.from_numpy(y)