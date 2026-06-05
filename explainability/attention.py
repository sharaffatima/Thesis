from matplotlib.axis import Axis

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
from models.anoformer import AnoFormerMainModel
from models.tranad import TranADMainModel
from utils.dataset import PTBXLTranADDataset


"""Attention Explainability and Visualization helpers."""

# ============================================================
# AnoFormer Explainability-Class
# ============================================================

class AnoFormerExplainability:
    """
    Erklärt AnoFormer-Entscheidungen mit vier Methoden:
      1. Attention-Heatmap   – wo schaut das Modell hin?
      3. Rekonstruktionsfehler-Plot – Original vs. Rekonstruktion + Score-Kurve
      4. Trainings-Verlaufsplot – Entwicklung aller Loss-Komponenten
    """

    def __init__(self, AnoFormerModel: AnoFormerMainModel):
        self.Model  = AnoFormerModel          # Referenz auf das Hauptmodell
        self.Device = AnoFormerModel.Device   # Gerät vom Modell übernehmen

    # ──────────────────────────────────────────────────────────
    def ShowAttentionHeatmap(
        self,
        TimeSeriesWindow: torch.Tensor,   # Ein einzelnes Fenster: (1, T, F)
        BlockIndex: int = 0,              # Welcher Encoder-Block soll visualisiert werden?
        HeadIndex: int  = 0,              # Welcher Attention-Kopf (0 bis NumHeads-1)?
        TitleSuffix: str = "",            # Optionaler Zusatz im Diagrammtitel
        UsePercentileScaling: bool = True,
        Percentile: float = 99.0,
    ) -> None:
        
        """
        Visualizes the attention matrix of one AnoFormer encoder block.

        Rows = query timesteps.
        Columns = key timesteps.
        Bright columns mean many timesteps attend to the same key timestep.
        """

        self.Model.Generator.eval() 
        Input = TimeSeriesWindow.to(self.Device) 

        with torch.no_grad(): 
            Embedded      = self.Model.Generator.InputEmbedding(Input) 
            WithPosition  = self.Model.Generator.PositionalEncoding(Embedded) 
            CurrentTensor = WithPosition
            for i, Block in enumerate(self.Model.Generator.EncoderStack.EncoderBlockList):
                CurrentTensor = Block(CurrentTensor) 
                if i == BlockIndex:
                    break  

        # Gespeicherte Attention-Gewichte aus dem Ziel-Block lesen: (1, H, T, T)
        AttWeights = self.Model.Generator.EncoderStack.EncoderBlockList[
            BlockIndex
        ].SelfAttention.StoredAttentionWeights

        if AttWeights is None:                           # Sicherheitscheck – Gewichte vorhanden?
            print("No attention weights available.")
            return

        AttMatrix = AttWeights[0, HeadIndex].detach().cpu().numpy() 

        print("Attention shape:", AttMatrix.shape)
        print("min:", AttMatrix.min())
        print("max:", AttMatrix.max())
        print("mean:", AttMatrix.mean())
        print("std:", AttMatrix.std())


        if UsePercentileScaling:
            vmax = np.percentile(AttMatrix, Percentile)
        else:
            vmax = AttMatrix.max()

        fig, Axis = plt.subplots(figsize=(9, 7))

        HeatmapImage = Axis.imshow(
            AttMatrix,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=vmax,
        )

        plt.colorbar(HeatmapImage, ax=Axis, label="Attention weight")
        Axis.set_xlabel("Key timestep")
        Axis.set_ylabel("Query timestep")
        Axis.set_title(
            f"Attention heatmap | Block {BlockIndex}, Head {HeadIndex} {TitleSuffix}"
        )

        plt.tight_layout()
        plt.show()





    # ──────────────────────────────────────────────────────────
    def GetAttentionImportance(
        self,
        TimeSeriesWindow: torch.Tensor,
        BlockIndex: int = 0,
        HeadIndex: int = 0,
    ) -> np.ndarray:
        
        """
        Returns mean attention received per timestep.
        Shape: (T,)
        """

        self.Model.Generator.eval()
        Input = TimeSeriesWindow.to(self.Device)

        with torch.no_grad():
            Embedded = self.Model.Generator.InputEmbedding(Input)
            WithPosition = self.Model.Generator.PositionalEncoding(Embedded)
            CurrentTensor = WithPosition

            for i, Block in enumerate(self.Model.Generator.EncoderStack.EncoderBlockList):
                CurrentTensor = Block(CurrentTensor)
                if i == BlockIndex:
                    break

        AttWeights = self.Model.Generator.EncoderStack.EncoderBlockList[
            BlockIndex
        ].SelfAttention.StoredAttentionWeights

        if AttWeights is None:
            raise ValueError("No attention weights available.")

        AttMatrix = AttWeights[0, HeadIndex].detach().cpu().numpy()

        # Mean attention received by each key timestep
        AttentionImportance = AttMatrix.mean(axis=0)

        return AttentionImportance
    
    # ──────────────────────────────────────────────────────────

    def GetReconstructionErrorOverTime(
        self,
        TimeSeriesWindow: torch.Tensor,
        DatasetForInverseNorm,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            Original data in original scale: (T, F)
            Reconstructed data in original scale: (T, F)
            Reconstruction error over time: (T,)
        """

        self.Model.Generator.eval()
        Input = TimeSeriesWindow.to(self.Device)

        with torch.no_grad():
            Masked, _ = self.Model.Masking(Input, None)
            Reconstructed = self.Model.Generator(Masked)

        OriginalNP = DatasetForInverseNorm.InverseNormalization(
            Input[0].detach().cpu().numpy()
        )

        ReconstructedNP = DatasetForInverseNorm.InverseNormalization(
            Reconstructed[0].detach().cpu().numpy()
        )

        ReconstructionError = ((OriginalNP - ReconstructedNP) ** 2).mean(axis=1)

        return OriginalNP, ReconstructedNP, ReconstructionError
    
    # ──────────────────────────────────────────────────────────
    def ShowAttentionAndReconstructionError(
        self,
        TimeSeriesWindow: torch.Tensor,
        DatasetForInverseNorm,
        BlockIndex: int = 0,
        HeadIndex: int = 0,
        TitleSuffix: str = "",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Plots attention importance and reconstruction error over time.

        Attention:
            Which timesteps were used as temporal context?

        Reconstruction error:
            Where did the model fail to reconstruct the ECG?
        """

        _, _, ReconstructionError = self.GetReconstructionErrorOverTime(
            TimeSeriesWindow=TimeSeriesWindow,
            DatasetForInverseNorm=DatasetForInverseNorm,
        )

        AttentionImportance = self.GetAttentionImportance(
            TimeSeriesWindow=TimeSeriesWindow,
            BlockIndex=BlockIndex,
            HeadIndex=HeadIndex,
        )

        ErrorScaled = ReconstructionError / (ReconstructionError.max() + 1e-8)
        AttentionScaled = AttentionImportance / (AttentionImportance.max() + 1e-8)

        plt.figure(figsize=(14, 4))

        plt.plot(ErrorScaled, label="Reconstruction error", linewidth=1.5)
        plt.plot(AttentionScaled, label="Attention importance", linewidth=1.5, alpha=0.8)

        plt.title(
            f"Attention vs. Reconstruction Error over Time {TitleSuffix}"
        )
        plt.xlabel("Timestep")
        plt.ylabel("Normalized value")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        return AttentionImportance, ReconstructionError


    # ──────────────────────────────────────────────────────────
    def ShowTrainingHistory(self) -> None:
        """
        Zeigt den GAN-Trainingsverlauf in zwei Teilbildern:
          Links:  Generator-Loss-Komponenten (Gesamt, GAN, Rekonstruktion)
          Rechts: Diskriminator-Loss
        """
        History = self.Model.TrainingHistory              # Verlaufsdaten holen
        if not any(History.values()):                     # Noch kein Training → Abbruch
            print("Kein Trainingsverlauf verfügbar.")
            return

        fig, Axes = plt.subplots(1, 2, figsize=(13, 4))   # Zwei nebeneinander liegende Plots

        Epochs = range(1, len(History["GeneratorTotalLoss"]) + 1)   # X-Achse: Epochennummern

        # ── Linkes Teilbild: Generator-Verluste ──────────────────
        Axes[0].plot(Epochs, History["GeneratorTotalLoss"], label="G Gesamt",        linewidth=1.5)
        Axes[0].plot(Epochs, History["GeneratorGANLoss"],   label="G GAN",           linewidth=1.2, linestyle="--")
        Axes[0].plot(Epochs, History["GeneratorRecLoss"],   label="G Rekonstruktion",linewidth=1.2, linestyle=":")
        Axes[0].set_title("Generator-Verlust")
        Axes[0].set_xlabel("Epoche")
        Axes[0].legend()
        Axes[0].grid(alpha=0.3)

        # ── Rechtes Teilbild: Diskriminator-Verlust ──────────────
        Axes[1].plot(Epochs, History["DiscriminatorLoss"], color="crimson", linewidth=1.5, label="D-Verlust")
        Axes[1].set_title("Diskriminator-Verlust")
        Axes[1].set_xlabel("Epoche")
        Axes[1].legend()
        Axes[1].grid(alpha=0.3)

        plt.suptitle("AnoFormer – Trainingsverlauf", fontweight="bold")  # Obertitel
        plt.tight_layout()
        plt.show()








# ============================================================
# TranADExplainability
# Erklärt Modellentscheidungen mit sechs Methoden:
# 1. ShowAttentionHeatmap     – Attention-Gewichte / Attention weights
# 2. ShowFeatureImportance    – Gradient×Input Saliency
# 3. ShowPhaseComparison      – Phase 1 vs. Phase 2 Rekonstruktion / reconstruction
# 4. ShowReconstructionError  – Original + Score + Klassifikation / classification
# 5. ShowTrainingHistory      – Loss und φ über Epochen / over epochs
# 6. ShowFocusScoreEffect     – Einfluss von φ / Effect of φ
# ============================================================

class TranADExplainability:
    """
    Erklärt TranAD-Anomalieentscheidungen mit sechs Visualisierungsmethoden.
    / Explains TranAD anomaly decisions with six visualization methods.
    """

    def __init__(self, MainModel: TranADMainModel):
        self.Model  = MainModel           # Referenz auf Hauptmodell / Reference to main model
        self.Device = MainModel.Device    # Gerät vom Modell übernehmen / Inherit device from model

    # ──────────────────────────────────────────────────────────────
    def ShowAttentionHeatmap(
        self,
        TimeSeriesWindow:  torch.Tensor,   # (1, T, F) – ein einzelnes Fenster / one single window
        LayerIndex:        int = 0,        # Schicht-Index / Layer index
        HeadIndex:         int = 0,        # Kopf-Index / Head index
        Component:         str = "Encoder",  # "Encoder", "Decoder1" oder "Decoder2" / "Encoder", "Decoder1" or "Decoder2"
    ) -> None:
        """
        Visualisiert Attention-Gewichte als Heatmap.
        / Visualizes attention weights as heatmap.
        Zeile i = Query-Token i schaut auf Spalte j = Key-Token j.
        / Row i = query token i attends to column j = key token j.
        """
        self.Model.Network.eval()
        InputTensor = TimeSeriesWindow.to(self.Device)

        with torch.no_grad():
            self.Model.Network(InputTensor)  # Forward-Pass füllt StoredAttentionWeights / Forward pass fills stored weights

        # Richtigen Attention-Modul auswählen / Select correct attention module
        if Component == "Encoder":
            LayerIdx = min(LayerIndex, len(self.Model.Network.EncoderStack.EncoderLayerList) - 1)
            AttModule = self.Model.Network.EncoderStack.EncoderLayerList[LayerIdx].SelfAttentionModule
            TitleStr  = f"Encoder Layer {LayerIdx} Self-Attention"
        elif Component == "Decoder1":
            LayerIdx = min(LayerIndex, len(self.Model.Network.DecoderStack1.DecoderLayerList) - 1)
            AttModule = self.Model.Network.DecoderStack1.DecoderLayerList[LayerIdx].CrossAttentionModule
            TitleStr  = f"Decoder1 Layer {LayerIdx} Cross-Attention"
        else:  # Decoder2
            LayerIdx = min(LayerIndex, len(self.Model.Network.DecoderStack2.DecoderLayerList) - 1)
            AttModule = self.Model.Network.DecoderStack2.DecoderLayerList[LayerIdx].CrossAttentionModule
            TitleStr  = f"Decoder2 Layer {LayerIdx} Cross-Attention"

        StoredWeights = AttModule.StoredAttentionWeights   # (1, H, T, T) oder None
        if StoredWeights is None:
            print("Keine gespeicherten Gewichte / No stored attention weights.")
            return

        AttMatrix = StoredWeights[0, HeadIndex].cpu().numpy()  # (T, T) – ein Kopf / one head

        fig, Axis = plt.subplots(figsize=(9, 7))
        HeatmapImg = Axis.imshow(AttMatrix, aspect="auto", cmap="viridis", interpolation="nearest")
        plt.colorbar(HeatmapImg, ax=Axis, label="Attention-Gewicht / Attention weight")
        Axis.set_xlabel("Key-Zeitschritt / Key timestep")
        Axis.set_ylabel("Query-Zeitschritt / Query timestep")
        Axis.set_title(f"TranAD Attention-Heatmap | {TitleStr}, Kopf/Head {HeadIndex}")
        plt.tight_layout()
        plt.show()

    # ──────────────────────────────────────────────────────────────
    def ShowFeatureImportance(
        self,
        TimeSeriesWindow:  torch.Tensor,                    # (1, T, F)
        FeatureNames:      Optional[List[str]] = None,     # Optionale Feature-Namen / Optional feature names
    ) -> np.ndarray:
        """
        Gradient×Input Feature-Wichtigkeit (Saliency-Methode).
        / Gradient×Input feature importance (saliency method).
        Formel / Formula: Wichtigkeit = |∂L/∂x · x|
        Hoher Wert = Feature hat starken Einfluss auf Phase-2-Fehler.
        / High value = feature strongly influences Phase-2 error.
        """
        self.Model.Network.eval()
        InputTensor = TimeSeriesWindow.to(self.Device).requires_grad_(True)  # Gradient auf Eingabe aktivieren / Enable gradient on input

        Reconstruction1, Reconstruction2 = self.Model.Network(InputTensor)  # Forward-Pass / Forward pass
        FeatureLoss = F.mse_loss(Reconstruction2, InputTensor.detach())      # Phase-2-Loss als Saliency-Loss / Phase-2 loss as saliency loss
        FeatureLoss.backward()                                                # Rückwärtsdurchlauf / Backward pass

        GradImportance   = (InputTensor.grad * InputTensor).abs()            # |∇·x| – element-weise / element-wise
        ImportanceMatrix = GradImportance[0].detach().cpu().numpy()          # (T, F) auf CPU / (T, F) on CPU

        fig, Axes = plt.subplots(1, 2, figsize=(14, 4))

        # Links: mittlere Wichtigkeit über Zeit / Left: mean importance over time
        Axes[0].plot(ImportanceMatrix.mean(axis=1), color="darkorange", linewidth=1.5)
        Axes[0].set_title("Mittlere Feature-Wichtigkeit über Zeit / Mean Feature Importance over Time")
        Axes[0].set_xlabel("Zeitschritt / Timestep")
        Axes[0].set_ylabel("Wichtigkeit / Importance")
        Axes[0].grid(alpha=0.3)

        # Rechts: Heatmap aller Features × Zeit / Right: heatmap of all features × time
        HeatmapImg = Axes[1].imshow(ImportanceMatrix.T, aspect="auto", cmap="hot", interpolation="nearest")
        plt.colorbar(HeatmapImg, ax=Axes[1], label="|Gradient × Input|")
        Axes[1].set_xlabel(" Timestep")
        Axes[1].set_ylabel(" Feature index")
        Axes[1].set_title(" Feature Importance Heatmap")
        if FeatureNames:                                    # Optionale Achsenbeschriftung / Optional axis labels
            Axes[1].set_yticks(range(len(FeatureNames)))
            Axes[1].set_yticklabels(FeatureNames, fontsize=8)
        plt.tight_layout()
        plt.show()
        return ImportanceMatrix                             # Matrix für weitere Analysen / Matrix for further analysis

    # ──────────────────────────────────────────────────────────────
    def ShowPhaseComparison(
        self,
        TimeSeriesWindow:      torch.Tensor,                    # (1, T, F)
        FeatureIndex:          int = 0,                        # Zu visualisierendes Feature / Feature to visualize
        DatasetForInverseNorm: Optional[PTBXLTranADDataset] = None, # Für Rücktransformation / For inverse transform
    ) -> None:
        """
        Vergleicht Phase-1- und Phase-2-Rekonstruktion mit dem Original.
        / Compares Phase-1 and Phase-2 reconstruction with the original.
        Zeigt: wie Phase-2 (adversariell) die Phase-1-Ausgabe verfeinert.
        / Shows: how Phase-2 (adversarial) refines Phase-1 output.
        """
        self.Model.Network.eval()
        with torch.no_grad():
            InputTensor = TimeSeriesWindow.to(self.Device)
            Rec1, Rec2  = self.Model.Network(InputTensor)   # Beide Phasen / Both phases

        # Numpy-Arrays für Plot vorbereiten / Prepare numpy arrays for plot
        Orig_np = InputTensor[0, :, FeatureIndex].cpu().numpy()  # Originalsequenz / Original sequence
        Rec1_np = Rec1[0, :, FeatureIndex].cpu().numpy()          # Phase-1-Rekonstruktion / Phase-1 reconstruction
        Rec2_np = Rec2[0, :, FeatureIndex].cpu().numpy()          # Phase-2-Rekonstruktion / Phase-2 reconstruction

        if DatasetForInverseNorm is not None:                    # Rücktransformation falls Dataset gegeben / Inverse transform if dataset given
            Orig_np = DatasetForInverseNorm.InverseNormalization(InputTensor[0].cpu().numpy())[:, FeatureIndex]
            Rec1_np = DatasetForInverseNorm.InverseNormalization(Rec1[0].cpu().numpy())[:, FeatureIndex]
            Rec2_np = DatasetForInverseNorm.InverseNormalization(Rec2[0].cpu().numpy())[:, FeatureIndex]

        TimeAxis = np.arange(len(Orig_np))  # Zeitachse / Time axis

        fig, Axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        # Oberes Panel: Original + beide Rekonstruktionen / Top: original + both reconstructions
        Axes[0].plot(TimeAxis, Orig_np, color="black",      linewidth=1.5, label="Original")
        Axes[0].plot(TimeAxis, Rec1_np, color="royalblue",  linewidth=1.2, linestyle="--", label="Phase 1 (x₁)")
        Axes[0].plot(TimeAxis, Rec2_np, color="crimson",    linewidth=1.2, linestyle=":",  label="Phase 2 (x₂)")
        Axes[0].set_title(f"Phase Comparison Feature {FeatureIndex}")
        Axes[0].legend()
        Axes[0].grid(alpha=0.3)

        # Unteres Panel: Absoluter Fehler beider Phasen / Bottom: absolute error of both phases
        Axes[1].fill_between(TimeAxis, np.abs(Orig_np - Rec1_np), alpha=0.5, color="royalblue", label="|Orig−Phase1|")
        Axes[1].fill_between(TimeAxis, np.abs(Orig_np - Rec2_np), alpha=0.5, color="crimson",   label="|Orig−Phase2|")
        Axes[1].set_title("Absolute Error")
        Axes[1].set_xlabel("Timestep")
        Axes[1].legend()
        Axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ──────────────────────────────────────────────────────────────
    def ShowReconstructionError(
        self,
        OriginalData:      np.ndarray,                         # (T, F) Originaldaten / Original data
        ReconstructedData: np.ndarray,                         # (T, F) Rekonstruktion Phase 2 / Phase-2 reconstruction
        AnomalyScores:     np.ndarray,                         # (N,) Anomalie-Scores / Anomaly scores
        TrueLabels:        Optional[np.ndarray] = None,        # (N,) Ground-Truth / Ground-Truth
        Threshold:         Optional[float] = None,             # Schwellenwert / Threshold
    ) -> None:
        """
        Drei-Panel-Visualisierung: Original, Score, Klassifikation.
        / Three-panel visualization: original, score, classification.
        """
        fig, Axes = plt.subplots(3, 1, figsize=(14, 10))  # 3 unabhängige Teilbilder / 3 independent panels

        TimeData  = np.arange(len(OriginalData))           # Zeitachse für Rohdaten / Time axis for raw data
        TimeScore = np.arange(len(AnomalyScores))          # Zeitachse für Scores / Time axis for scores

        # Panel 1: Original vs. Rekonstruktion / Original vs. reconstruction
        for f in range(min(3, OriginalData.shape[1])):     # Maximal 3 Features / Maximum 3 features
            Axes[0].plot(TimeData, OriginalData[:, f],      linewidth=0.8, label=f"Orig F{f}")
            Axes[0].plot(TimeData, ReconstructedData[:, f], linewidth=0.8, linestyle="--", alpha=0.7, label=f"Rek F{f}")
        Axes[0].set_title("Original vs. Phase-2 Reconstruction")
        Axes[0].legend(fontsize=7, ncol=4)
        Axes[0].grid(alpha=0.3)

        # Panel 2: Anomalie-Score / Anomaly score
        Axes[1].fill_between(TimeScore, AnomalyScores, alpha=0.5, color="crimson", label=" Anomaly Score")
        Axes[1].plot(TimeScore, AnomalyScores, linewidth=0.8, color="crimson")
        if Threshold is not None:                          # Schwellenwert einzeichnen / Draw threshold
            Axes[1].axhline(Threshold, color="black", linestyle="--", linewidth=1.2,
                            label=f"SThreshold={Threshold:.5f}")
        Axes[1].set_title("Anomaly Score (MSE Phase 2)")
        Axes[1].legend(fontsize=8)
        Axes[1].grid(alpha=0.3)

        # Panel 3: Klassifikation / Classification
        if Threshold is not None:
            PredLabels = (AnomalyScores > Threshold).astype(int)  # Binäre Vorhersage / Binary prediction
            Axes[2].fill_between(TimeScore, PredLabels, alpha=0.6, color="darkorange", label=" Predicted")
        if TrueLabels is not None:
            N = min(len(TrueLabels), len(AnomalyScores))
            Axes[2].fill_between(np.arange(N), TrueLabels[:N] * 0.5,
                                 alpha=0.5, color="steelblue", label="True (×0.5)")
        Axes[2].set_title(" Classification")
        Axes[2].set_xlabel(" Window index")
        Axes[2].legend(fontsize=8)
        Axes[2].grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ──────────────────────────────────────────────────────────────
    def ShowTrainingHistory(self) -> None:
        """
        Visualisiert Verlauf von Gesamtverlust und φ über alle Epochen.
        / Visualizes history of total loss and φ over all epochs.
        """
        History = self.Model.TrainingHistory
        if not History["TotalLoss"]:                       # Noch kein Training / No training yet
            print("No training history available.")
            return

        Epochs = range(1, len(History["TotalLoss"]) + 1)  # Epochen-Achse / Epoch axis

        fig, Axes = plt.subplots(1, 2, figsize=(13, 4))

        # Links: Loss / Left: Loss
        Axes[0].plot(Epochs, History["TotalLoss"], linewidth=1.5, color="steelblue", label="TranAD Loss")
        Axes[0].set_title(" Training Loss")
        Axes[0].set_xlabel("Epoch")
        Axes[0].set_ylabel("Loss")
        Axes[0].legend()
        Axes[0].grid(alpha=0.3)

        # Rechts: φ / Right: φ
        Axes[1].plot(Epochs, History["FocusScore"], linewidth=1.5, color="darkorange", label="φ (Focus Score)")
        Axes[1].set_title("Focus Score φ per Epoch")
        Axes[1].set_xlabel(" Epoch")
        Axes[1].set_ylabel("φ")
        Axes[1].legend()
        Axes[1].grid(alpha=0.3)

        plt.suptitle("TranAD – Training History", fontweight="bold")
        plt.tight_layout()
        plt.show()

    # ──────────────────────────────────────────────────────────────
    def ShowFocusScoreEffect(
        self,
        TimeSeriesWindow: torch.Tensor,        # (1, T, F)
        FocusScoreList:   Optional[List[float]] = None,  # Zu vergleichende φ-Werte / φ values to compare
    ) -> None:
        """
        Zeigt wie verschiedene φ-Werte den kombinierten Loss gewichten.
        / Shows how different φ values weight the combined loss.
        L(φ) = (1/φ)·E1 + φ·E2 – für jedes φ eine Kurve / one curve per φ.
        """
        if FocusScoreList is None:
            FocusScoreList = [0.5, 1.0, 2.0, 5.0, 10.0, 30.0]  # Standard-Vergleichswerte / Default values

        self.Model.Network.eval()
        with torch.no_grad():
            InputTensor = TimeSeriesWindow.to(self.Device)
            Rec1, Rec2  = self.Model.Network(InputTensor)
            Error1 = ((Rec1 - InputTensor) ** 2).mean(dim=-1)[0].cpu().numpy()  # (T,) Phase-1-Fehler / Phase-1 error
            Error2 = ((Rec2 - InputTensor) ** 2).mean(dim=-1)[0].cpu().numpy()  # (T,) Phase-2-Fehler / Phase-2 error

        TimeAxis = np.arange(len(Error1))  # Zeitachse / Time axis

        fig, Axes = plt.subplots(1, 2, figsize=(14, 5))

        # Links: Rohfehler beider Phasen / Left: raw errors of both phases
        Axes[0].plot(TimeAxis, Error1, color="royalblue", linewidth=1.5, label=" Phase-1 error")
        Axes[0].plot(TimeAxis, Error2, color="crimson",   linewidth=1.5, label="P Phase-2 error")
        Axes[0].set_title(" Raw Errors Phase 1 vs. Phase 2")
        Axes[0].set_xlabel("Timestep")
        Axes[0].legend()
        Axes[0].grid(alpha=0.3)

        # Rechts: kombinierter Loss für jedes φ / Right: combined loss for each φ
        for Phi in FocusScoreList:
            CombinedLoss = Error1 / max(Phi, 1e-6) + Error2 * Phi  # L(φ) = E1/φ + φ·E2
            Axes[1].plot(TimeAxis, CombinedLoss, linewidth=1.2, label=f"φ={Phi:.1f}")
        Axes[1].set_title(" Combined TranAD Loss for various φ")
        Axes[1].set_xlabel("Timestep")
        Axes[1].set_ylabel("L = E₁/φ + φ·E₂")
        Axes[1].legend(fontsize=8)
        Axes[1].grid(alpha=0.3)

        plt.suptitle(" Effect of Focus Score φ", fontweight="bold")
        plt.tight_layout()
        plt.show()
