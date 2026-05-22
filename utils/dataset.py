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
# TranADDataset & TranADDataLoader
# Modulares Datenmanagement: Fenstererzeugung, Normalisierung
# ============================================================

class TranADDataset(Dataset):
    """
    PyTorch-Dataset für TranAD: erstellt überlappende Fenster aus einer Zeitreihe.
    Unterstützt Z-Score- und Min-Max-Normalisierung.
    """

    def __init__(
        self,
        TimeSeriesData:    np.ndarray,                    # (T, F) – Rohdaten / Raw data
        WindowSize:        int = 64,                      # Fensterlänge / Window length
        Stride:            int = 1,                       # Schrittweite / Step size
        NormalizationType: Optional[str] = "zscore",                 # "zscore" oder "minmax" / "zscore" or "minmax"
        Labels:            Optional[np.ndarray] = None,  # (T,) Anomalie-Labels (0/1) / Anomaly labels
    ):
        super().__init__()
        self.WindowSize        = WindowSize         # Fensterbreite speichern / Store window width
        self.Stride            = Stride             # Schrittweite speichern / Store stride
        self.NormalizationType = NormalizationType  # Normalisierungstyp / Normalization type

        # Normalisierungsparameter (gesetzt in _NormalizeData) / Normalization params (set in _NormalizeData)
        self.NormalizationMean: Optional[np.ndarray] = None  # (F,) – Z-Score Mittelwert / Z-Score mean
        self.NormalizationStd:  Optional[np.ndarray] = None  # (F,) – Z-Score Std.-Abw. / Z-Score std
        self.NormalizationMin:  Optional[np.ndarray] = None  # (F,) – Min-Max Minimum / Min-Max minimum
        self.NormalizationMax:  Optional[np.ndarray] = None  # (F,) – Min-Max Maximum / Min-Max maximum

        NormalizedData = self._NormalizeData(TimeSeriesData)  # Normalisieren / Normalize
        TotalLength    = len(NormalizedData)                  # Zeitreihen-Länge / Time series length

        # ── Fenster und Labels erzeugen / Create windows and labels ──
        self.Windows:      List[np.ndarray] = []   # Liste aller Fenster-Arrays / List of all window arrays
        self.WindowLabels: List[int]        = []   # Zugehörige binäre Labels / Corresponding binary labels

        for StartIdx in range(0, TotalLength - WindowSize + 1, Stride):
            EndIdx = StartIdx + WindowSize                          # End-Index des Fensters / Window end index
            self.Windows.append(NormalizedData[StartIdx:EndIdx].copy())  # Fenster speichern / Store window
            if Labels is not None:
                WindowLabel = int(Labels[StartIdx:EndIdx].max())    # 1 wenn irgendein Punkt anomal / 1 if any anomalous
            else:
                WindowLabel = 0                                     # Kein Label → als normal behandeln / No label → treat as normal
            self.WindowLabels.append(WindowLabel)

        print(
            f"TranADDataset: {len(self.Windows)} Fenster aus {TotalLength} Zeitpunkten "
            f"(W={WindowSize}, S={Stride}) / "
            f"{len(self.Windows)} windows from {TotalLength} timesteps"
        )

    def _NormalizeData(self, Data: np.ndarray) -> np.ndarray:
        """Normalisiert und speichert Parameter / Normalizes and stores parameters."""
        if self.NormalizationType == "zscore":
            self.NormalizationMean = Data.mean(axis=0)             # Spaltenmittelwerte / Column means
            self.NormalizationStd  = Data.std(axis=0) + 1e-8       # Std + ε gegen Division durch Null / std + ε
            return ((Data - self.NormalizationMean) / self.NormalizationStd).astype(np.float32)
        elif self.NormalizationType == "minmax":
            self.NormalizationMin = Data.min(axis=0)               # Spalten-Minima / Column minima
            self.NormalizationMax = Data.max(axis=0)               # Spalten-Maxima / Column maxima
            Range = self.NormalizationMax - self.NormalizationMin + 1e-8  # Bereich + ε / Range + ε
            return ((Data - self.NormalizationMin) / Range).astype(np.float32)
        else:
            raise ValueError(f"Unknown normalization type: {self.NormalizationType}")

    def InverseNormalization(self, NormalizedData: np.ndarray) -> np.ndarray:
        """Macht Normalisierung rückgängig / Reverses normalization."""
        if self.NormalizationType == "zscore" and self.NormalizationMean is not None:
            return NormalizedData * self.NormalizationStd + self.NormalizationMean  # Rücktransformation Z-Score
        elif self.NormalizationType == "minmax" and self.NormalizationMin is not None:
            Range = self.NormalizationMax - self.NormalizationMin + 1e-8
            return NormalizedData * Range + self.NormalizationMin               # Rücktransformation Min-Max
        return NormalizedData  # Keine Normalisierung aktiv / No normalization active

    def __len__(self) -> int:
        return len(self.Windows)                           # Anzahl Fenster / Number of windows

    def __getitem__(self, Index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        WindowTensor = torch.tensor(self.Windows[Index],      dtype=torch.float32)  # Fenster als Float-Tensor
        LabelTensor  = torch.tensor(self.WindowLabels[Index], dtype=torch.long)     # Label als Long-Tensor
        return WindowTensor, LabelTensor                   # (T, F), Label


# ─────────────────────────────────────────────────────────────────────
class TranADDataLoader:
    """
    Verwaltet Training- und Test-DataLoader für TranAD.
    Kapselt TranADDataset und torch.utils.data.DataLoader.
    """

    def __init__(
        self,
        TrainingData:      np.ndarray,                    # (T_train, F) Trainingsdaten / Training data
        TestData:          np.ndarray,                    # (T_test, F) Testdaten / Test data
        WindowSize:        int   = 64,                    # Fensterlänge / Window size
        TrainingStride:    int   = 4,                     # Stride beim Training / Training stride
        TestStride:        Optional[int]   = None,        # Stride beim Test (None → WindowSize) / Test stride
        NormalizationType: Optional[str]   = "zscore",              # Normalisierungstyp / Normalization type
        BatchSize:         int   = 32,                    # Batch-Größe / Batch size
        TestLabels:        Optional[np.ndarray] = None,  # (T_test,) Ground-Truth Labels / Ground-Truth labels
    ):
        ActualTestStride = TestStride if TestStride is not None else WindowSize  # Kein Überlapp beim Test / No overlap in test

        # ── Datasets erstellen / Create datasets ──────────────────
        self.TrainingDataset = TranADDataset(
            TrainingData, WindowSize, TrainingStride, NormalizationType
        )                                                 # Training: überlappende Fenster / Overlapping windows
        self.TestDataset = TranADDataset(
            TestData, WindowSize, ActualTestStride, NormalizationType, Labels=TestLabels
        )                                                 # Test: kein Überlapp + optionale Labels / No overlap + optional labels

        self.BatchSize   = BatchSize                      # Batch-Größe merken / Store batch size
        self.WindowSize  = WindowSize                     # Fensterlänge merken / Store window size

        # ── DataLoader erstellen / Create DataLoaders ─────────────
        self.TrainingLoader = DataLoader(
            self.TrainingDataset,
            batch_size  = BatchSize,
            shuffle     = True,    # Reihenfolge beim Training mischen / Shuffle order during training
            drop_last   = True,    # Unvollständige Batches verwerfen / Drop incomplete batches
            num_workers = 0,       # Keine Parallelprozesse (macOS-kompatibel) / No subprocesses
        )
        self.TestLoader = DataLoader(
            self.TestDataset,
            batch_size  = BatchSize,
            shuffle     = False,   # Reihenfolge beim Test beibehalten / Keep order during testing
            drop_last   = False,   # Alle Test-Fenster behalten / Keep all test windows
            num_workers = 0,
        )

        print(
            f"TranADDataLoader: "
            f"Train-Batches={len(self.TrainingLoader)} | "
            f"Test-Batches={len(self.TestLoader)} | "
            f"Batch size={BatchSize}"
        )


