import torch
from torch.utils.data import Dataset


class ECGReconstructionDataset(Dataset):
    def __init__(self, X, y=None):
        """
        X shape: [num_samples, time_steps, num_leads]
        Example: [1000, 1000, 12]

        y shape: [num_samples]
        0 = normal, 1 = anomaly
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]

        # training: no labels needed
        if self.y is None:
            return x

        # evaluation: keep label too
        return x, self.y[idx]