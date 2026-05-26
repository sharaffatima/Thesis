import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List


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
    


# ============================================================
# AnoFormerDataset (PyTorch Dataset-Klasse)
# Verwaltet Laden, Normalisierung und Fenstersegmentierung
# ============================================================

class AnoFormerDataset(Dataset):
    """
    PyTorch-kompatibler Datensatz für Zeitreihendaten.
    Unterstützt: Z-Score- und Min-Max-Normalisierung, Sliding-Window-Segmentierung,
    optionale Anomalie-Labels, Speichern/Laden.
    """

    def __init__(
        self,
        TimeSeriesData: np.ndarray,            # Rohdaten: (Gesamtzeit, AnzahlFeatures)
        WindowSize: int = 64,                  # Länge eines Fensters (Anzahl Zeitschritte)
        Stride: int = 1,                       # Schrittweite zwischen aufeinanderfolgenden Fenstern
        NormalizationType: str = "zscore",     # Normalisierungsart: "zscore" oder "minmax"
        Labels: np.ndarray | None = None,      # Optionale Anomalie-Labels (0=normal, 1=anomal)
    ):
        super().__init__()
        self.WindowSize        = WindowSize         # Fenstergröße merken
        self.Stride            = Stride             # Schrittweite merken
        self.NormalizationType = NormalizationType  # Normalisierungstyp merken

        # ── Normalisierungsparameter aus den Rohdaten berechnen ──
        self.Mean              = TimeSeriesData.mean(axis=0)          # Spaltenmittelwert (pro Feature)
        self.StandardDeviation = TimeSeriesData.std(axis=0) + 1e-8    # Spalten-Std.-abw. + ε (kein Div/0)
        self.Minimum           = TimeSeriesData.min(axis=0)           # Spaltenminimum
        self.Maximum           = TimeSeriesData.max(axis=0)           # Spaltenmaximum
        self.Range             = self.Maximum - self.Minimum + 1e-8   # Spannweite (+ ε gegen Div/0)

        # ── Normalisierung anwenden und als float32 speichern ────
        NormalizedData      = self._Normalize(TimeSeriesData)         # Daten normieren
        self.NormalizedData = NormalizedData.astype(np.float32)       # float32 für PyTorch-Kompatibilität

        # ── Labels vorbereiten ────────────────────────────────────
        self.HasLabels = Labels is not None                            # Merken ob Labels vorhanden sind
        if self.HasLabels:
            self.Labels = Labels.astype(np.int64)                      # Echte Labels als int64
        else:
            self.Labels = np.zeros(len(TimeSeriesData), dtype=np.int64)  # Platzhalter-Labels (alle 0)

        # ── Fensterindizes vorberechnen ────────────────────────────
        self.WindowIndices = self._GenerateWindows()                   # Liste der gültigen Startindizes

    def _Normalize(self, Data: np.ndarray) -> np.ndarray:
        """Normalisiert die Daten entsprechend dem gewählten Typ."""
        if self.NormalizationType == "zscore":
            return (Data - self.Mean) / self.StandardDeviation   # Z-Score: Mittelwert 0, Std.-abw. 1
        elif self.NormalizationType == "minmax":
            return (Data - self.Minimum) / self.Range            # Min-Max: skaliert auf [0, 1]
        else:
            raise ValueError(f"Unbekannter NormalizationType: {self.NormalizationType}")

    def InverseNormalization(self, NormalizedData: np.ndarray) -> np.ndarray:
        """Rekonstruiert die Originalskala aus normierten Daten (für Visualisierung)."""
        if self.NormalizationType == "zscore":
            return NormalizedData * self.StandardDeviation + self.Mean  # Rücktransformation Z-Score
        elif self.NormalizationType == "minmax":
            return NormalizedData * self.Range + self.Minimum           # Rücktransformation Min-Max

    def _GenerateWindows(self) -> list[int]:
        """Berechnet alle gültigen Fenster-Startindizes (Sliding Window)."""
        TotalLength  = len(self.NormalizedData)                        # Gesamtlänge der Zeitreihe
        # range(0, Ende, Stride) → jeder Schritt um Stride versetzt
        StartIndices = list(range(0, TotalLength - self.WindowSize + 1, self.Stride))
        return StartIndices                                            # Liste der Startpositionen

    def __len__(self) -> int:
        return len(self.WindowIndices)                                 # Anzahl der verfügbaren Fenster

    def __getitem__(self, Index: int) -> tuple[torch.Tensor, torch.Tensor]:
        Start       = self.WindowIndices[Index]                        # Startindex des Fensters
        End         = Start + self.WindowSize                          # Endindex (exklusiv)
        WindowData  = self.NormalizedData[Start:End]                   # Daten-Slice: (W, F)
        WindowLabel = self.Labels[Start:End]                           # Label-Slice: (W,)
        # NumPy-Arrays in PyTorch-Tensoren umwandeln
        return torch.from_numpy(WindowData), torch.from_numpy(WindowLabel)

    def CreateDataLoader(
        self, BatchSize: int = 32, Shuffle: bool = True, NumWorkers: int = 0
    ) -> DataLoader:
        """Erstellt einen fertig konfigurierten PyTorch DataLoader."""
        return DataLoader(
            self,                                       # Dieses Dataset als Datenquelle
            batch_size=BatchSize,                       # Anzahl Fenster pro Batch
            shuffle=Shuffle,                            # Beim Training mischen, beim Test nicht
            num_workers=NumWorkers,                     # Anzahl paralleler Datenlader-Prozesse
            pin_memory=(DEVICE.type == "cuda"),         # Pinned Memory → schnellerer GPU-Transfer
            drop_last=False,                            # Letzten Batch auch verwenden (auch wenn kleiner)
        )

    def Save(self, Path: str) -> None:
        """Speichert normalisierte Daten und alle Normalisierungsparameter als .npz-Datei."""
        np.savez(
            Path,                                       # Dateipfad (ohne .npz – wird automatisch ergänzt)
            NormalizedData=self.NormalizedData,         # Normierte Zeitreihendaten
            Labels=self.Labels,                         # Anomalie-Labels
            Mean=self.Mean,                             # Mittelwert pro Feature
            StandardDeviation=self.StandardDeviation,   # Standardabweichung pro Feature
            Minimum=self.Minimum,                       # Minimum pro Feature
            Maximum=self.Maximum,                       # Maximum pro Feature
        )
        print(f"Dataset gespeichert unter: {Path}.npz")

    @classmethod
    def Load(
        cls,
        Path: str,
        WindowSize: int = 64,
        Stride: int = 1,
        NormalizationType: str = "zscore",
    ) -> "AnoFormerDataset":
        """Lädt einen zuvor gespeicherten Datensatz aus einer .npz-Datei."""
        Saved    = np.load(Path, allow_pickle=True)     # Datei öffnen und alle Arrays laden
        Instance = cls.__new__(cls)                      # Instanz ohne __init__ erzeugen
        super(AnoFormerDataset, Instance).__init__()     # Dataset-Elternklasse initialisieren
        # ── Alle gespeicherten Attribute wiederherstellen ─────────
        Instance.WindowSize        = WindowSize
        Instance.Stride            = Stride
        Instance.NormalizationType = NormalizationType
        Instance.NormalizedData    = Saved["NormalizedData"]         # Normierte Daten
        Instance.Labels            = Saved["Labels"]                 # Labels
        Instance.Mean              = Saved["Mean"]                   # Mittelwert
        Instance.StandardDeviation = Saved["StandardDeviation"]      # Std.-abweichung
        Instance.Minimum           = Saved["Minimum"]                # Minimum
        Instance.Maximum           = Saved["Maximum"]                # Maximum
        Instance.Range             = Instance.Maximum - Instance.Minimum + 1e-8  # Spannweite neu berechnen
        Instance.HasLabels         = Instance.Labels.any()           # Gibt es Labels ≠ 0?
        Instance.WindowIndices     = Instance._GenerateWindows()     # Fensterindizes regenerieren
        print(f"Dataset geladen von: {Path}")
        return Instance                                              # Fertig geladene Instanz zurückgeben






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

