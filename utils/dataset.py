import torch
import numpy as np
from torch.utils.data import Dataset

# ============================================================
# AnoFormer Dataset: PTBXLAnoFormerDataset
# ============================================================

class PTBXLAnoFormerDataset(Dataset):
    """
    Dataset for PTB-XL ECG records for AnoFormer.

    Input X:
        [num_records, 1000, 12]

    Label y:
        [num_records]
        0 = normal
        1 = abnormal
    """

    def __init__(
        self,
        X,
        y=None,
        mean=None,
        std=None,
        normalize=True,
    ):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64) if y is not None else np.zeros(len(X), dtype=np.int64)

        self.normalize = normalize

        if normalize:
            if mean is None or std is None:
                self.mean = self.X.mean(axis=(0, 1), keepdims=True)
                self.std = self.X.std(axis=(0, 1), keepdims=True) + 1e-8
            else:
                self.mean = mean
                self.std = std

            self.X = (self.X - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)  # (1000, 12)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

    def InverseNormalization(self, X):
        if not self.normalize:
            return X

        return X * self.std.squeeze() + self.mean.squeeze()



# ============================================================
# TranAD Dataset: PTBXLTranADDataset
# ============================================================

class PTBXLTranADDataset(Dataset):
    """
    Dataset for PTB-XL ECG records.

    Expected X shape:
        [num_records, time_steps, num_leads]
        example: [1000, 1000, 12]

    Expected y shape:
        [num_records]
        0 = normal, 1 = anomaly
    """

    def __init__(self, X, y=None, mean=None, std=None, normalize=True):
        self.X = X.astype(np.float32)
        self.y = y

        self.normalize = normalize

        if normalize:
            if mean is None or std is None:
                # Compute only on training data
                self.mean = self.X.mean(axis=(0, 1), keepdims=True)
                self.std = self.X.std(axis=(0, 1), keepdims=True) + 1e-8
            else:
                self.mean = mean
                self.std = std

            self.X = (self.X - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)

        if self.y is None:
            return x, torch.tensor(0, dtype=torch.long)

        label = torch.tensor(self.y[idx], dtype=torch.long)
        return x, label

    def InverseNormalization(self, X):
      if not self.normalize:
        return X

      return X * self.std.squeeze() + self.mean.squeeze()

