import torch
import numpy as np
from torch.utils.data import Dataset


# ============================================================
# PTB-XL ECG Dataset
# ============================================================

class PTBXLDataset(Dataset):
    """
    Shared dataset for PTB-XL ECG records.

    Input X:
        shape (num_records, 1000, 12)

    Label y:
        shape (num_records,)
        0 = normal
        1 = anomalous

    The dataset applies per-lead standardization using mean and
    standard deviation computed from the training data.
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

        if y is None:
            self.y = np.zeros(
                len(X),
                dtype=np.int64,
            )
        else:
            self.y = y.astype(np.int64)

        self.normalize = normalize

        if normalize:
            if mean is None or std is None:
                self.mean = self.X.mean(
                    axis=(0, 1),
                    keepdims=True,
                )

                self.std = (
                    self.X.std(
                        axis=(0, 1),
                        keepdims=True,
                    )
                    + 1e-8
                )

            else:
                self.mean = mean
                self.std = std

            self.X = (
                self.X
                - self.mean
            ) / self.std

        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(
            self.X[idx],
            dtype=torch.float32,
        )

        y = torch.tensor(
            self.y[idx],
            dtype=torch.long,
        )

        return x, y

    def InverseNormalization(self, X):
        """
        Converts normalized ECG data back to the original scale.

        Input:
            X: shape (1000, 12) or compatible NumPy array
        """

        if not self.normalize:
            return X

        return (
            X * self.std.squeeze()
            + self.mean.squeeze()
        )