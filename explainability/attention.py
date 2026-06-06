import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Optional, List
from models.anoformer import AnoFormerMainModel
from models.tranad import TranADMainModel
from utils.dataset import PTBXLTranADDataset


"""Attention Explainability and Visualization helpers."""

# ============================================================
# AnoFormer Attention Explainability
# ============================================================

class AnoFormerExplainability:
    """
    Attention-based explainability for AnoFormer on ECG data.

    Purpose:
      1. Compare original and reconstructed ECG signals.
      2. Visualize temporal self-attention.
      3. Compare attention importance with reconstruction error.

    Important:
      Attention explains temporal focus, not lead importance.
      Reconstruction error is the direct basis of the anomaly score.
    """

    def __init__(self, AnoFormerModel):
        self.Model = AnoFormerModel
        self.Device = AnoFormerModel.Device

    # ──────────────────────────────────────────────────────────
    def _get_attention_matrix(
        self,
        TimeSeriesWindow: torch.Tensor,
        BlockIndex: int = 0,
        HeadIndex: int | None = None,
    ) -> np.ndarray:
        """
        Returns attention matrix of shape (T, T).

        If HeadIndex is None, attention is averaged across all heads.
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

        # AttWeights shape: (1, heads, T, T)
        if HeadIndex is None:
            AttMatrix = AttWeights[0].mean(dim=0).detach().cpu().numpy()
        else:
            AttMatrix = AttWeights[0, HeadIndex].detach().cpu().numpy()

        return AttMatrix

    # ──────────────────────────────────────────────────────────
    def GetAttentionImportance(
        self,
        TimeSeriesWindow: torch.Tensor,
        BlockIndex: int = 0,
        HeadIndex: int | None = None,
    ) -> np.ndarray:
        """
        Returns mean attention received per timestep.
        Shape: (T,)
        """

        AttMatrix = self._get_attention_matrix(
            TimeSeriesWindow=TimeSeriesWindow,
            BlockIndex=BlockIndex,
            HeadIndex=HeadIndex,
        )

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
            Original data: (T, F)
            Reconstructed data: (T, F)
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

        # Mean squared error across all 12 leads per timestep
        ReconstructionError = ((OriginalNP - ReconstructedNP) ** 2).mean(axis=1)

        return OriginalNP, ReconstructedNP, ReconstructionError

    # ──────────────────────────────────────────────────────────
    def CompareNormalAndAnomalousAttention(
        self,
        NormalWindow: torch.Tensor,
        AnomalousWindow: torch.Tensor,
        DatasetForInverseNorm,
        NormalIndex: int,
        AnomalousIndex: int,
        LeadIndex: int = 1,
        LeadName: str = "II",
        BlockIndex: int = 0,
        HeadIndex: int | None = None,
        Percentile: float = 99.0,
    ) -> None:
        """
        Compares attention and reconstruction behavior for one normal
        and one anomalous ECG.

        Rows:
          1. Original vs reconstructed signal for one ECG lead.
          2. Attention heatmap.
          3. Attention importance vs reconstruction error.
        """

        def compute_all(Window):
            OriginalNP, ReconstructedNP, ReconstructionError = (
                self.GetReconstructionErrorOverTime(
                    TimeSeriesWindow=Window,
                    DatasetForInverseNorm=DatasetForInverseNorm,
                )
            )

            AttMatrix = self._get_attention_matrix(
                TimeSeriesWindow=Window,
                BlockIndex=BlockIndex,
                HeadIndex=HeadIndex,
            )

            AttentionImportance = AttMatrix.mean(axis=0)

            return OriginalNP, ReconstructedNP, ReconstructionError, AttMatrix, AttentionImportance

        normal_original, normal_rec, normal_error, normal_att, normal_att_imp = compute_all(NormalWindow)
        anom_original, anom_rec, anom_error, anom_att, anom_att_imp = compute_all(AnomalousWindow)

        time_axis = np.arange(normal_original.shape[0])

        # Shared heatmap scale for fair comparison
        shared_vmax = max(
            np.percentile(normal_att, Percentile),
            np.percentile(anom_att, Percentile),
        )

        if HeadIndex is None:
            attention_title = f"Block {BlockIndex}, all heads averaged"
        else:
            attention_title = f"Block {BlockIndex}, Head {HeadIndex}"

        fig, axes = plt.subplots(3, 2, figsize=(20, 18))



        # ── Row 1: Original vs reconstruction, grouped ECG leads ─────
        ECG_LEADS = [
            "I", "II", "III",
            "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6"
        ]

        LeadGroups = [
            [0, 1, 2],      # Limb leads
            [3, 4, 5],      # Augmented leads
            [6, 7, 8],      # Precordial V1-V3
            [9, 10, 11],    # Precordial V4-V6
        ]

        GroupTitles = [
            "Leads I, II, III",
            "Leads aVR, aVL, aVF",
            "Leads V1, V2, V3",
            "Leads V4, V5, V6",
        ]

        # Create nested subplots inside top-left and top-right cells
        normal_subgrid = axes[0, 0].get_subplotspec().subgridspec(4, 1, hspace=0.35)
        anom_subgrid = axes[0, 1].get_subplotspec().subgridspec(4, 1, hspace=0.35)

        axes[0, 0].remove()
        axes[0, 1].remove()

        normal_lead_axes = [fig.add_subplot(normal_subgrid[i, 0]) for i in range(4)]
        anom_lead_axes = [fig.add_subplot(anom_subgrid[i, 0]) for i in range(4)]

        for group_idx, lead_group in enumerate(LeadGroups):
            ax_n = normal_lead_axes[group_idx]
            ax_a = anom_lead_axes[group_idx]

            for lead_idx in lead_group:
                lead_name = ECG_LEADS[lead_idx]

                ax_n.plot(
                    time_axis,
                    normal_original[:, lead_idx],
                    linewidth=0.8,
                    label=f"{lead_name} original",
                )
                ax_n.plot(
                    time_axis,
                    normal_rec[:, lead_idx],
                    linestyle="--",
                    linewidth=0.8,
                    label=f"{lead_name} reconstructed",
                )

                ax_a.plot(
                    time_axis,
                    anom_original[:, lead_idx],
                    linewidth=0.8,
                    label=f"{lead_name} original",
                )
                ax_a.plot(
                    time_axis,
                    anom_rec[:, lead_idx],
                    linestyle="--",
                    linewidth=0.8,
                    label=f"{lead_name} reconstructed",
                )

            ax_n.set_title(f"Normal | {GroupTitles[group_idx]}", fontsize=9)
            ax_n.grid(alpha=0.3)
            ax_n.legend(fontsize=6, ncol=3)

            ax_a.set_title(f"Anomalous | {GroupTitles[group_idx]}", fontsize=9)
            ax_a.grid(alpha=0.3)
            ax_a.legend(fontsize=6, ncol=3)

        normal_lead_axes[-1].set_xlabel("Timestep")
        anom_lead_axes[-1].set_xlabel("Timestep")
        normal_lead_axes[0].set_ylabel(f"Sample {NormalIndex}")
        anom_lead_axes[0].set_ylabel(f"Sample {AnomalousIndex}")

        # ── Row 2: Attention heatmaps ─────────────────────────────
        im0 = axes[1, 0].imshow(
            normal_att,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=shared_vmax,
        )
        axes[1, 0].set_title(f"Attention Heatmap | Normal | {attention_title}")
        axes[1, 0].set_xlabel("Key timestep")
        axes[1, 0].set_ylabel("Query timestep")
        fig.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im1 = axes[1, 1].imshow(
            anom_att,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=shared_vmax,
        )
        axes[1, 1].set_title(f"Attention Heatmap | Anomalous | {attention_title}")
        axes[1, 1].set_xlabel("Key timestep")
        axes[1, 1].set_ylabel("Query timestep")
        fig.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # ── Row 3: Attention vs reconstruction error ──────────────
        normal_error_scaled = normal_error / (normal_error.max() + 1e-8)
        normal_att_scaled = normal_att_imp / (normal_att_imp.max() + 1e-8)

        anom_error_scaled = anom_error / (anom_error.max() + 1e-8)
        anom_att_scaled = anom_att_imp / (anom_att_imp.max() + 1e-8)

        axes[2, 0].plot(
            time_axis,
            normal_error_scaled,
            label="Reconstruction error",
            linewidth=1.2,
        )
        axes[2, 0].plot(
            time_axis,
            normal_att_scaled,
            label="Attention importance",
            linewidth=1.2,
            alpha=0.8,
        )
        axes[2, 0].set_title("Attention vs Reconstruction Error | Normal")
        axes[2, 0].set_xlabel("Timestep")
        axes[2, 0].set_ylabel("Normalized value")
        axes[2, 0].legend()
        axes[2, 0].grid(alpha=0.3)

        axes[2, 1].plot(
            time_axis,
            anom_error_scaled,
            label="Reconstruction error",
            linewidth=1.2,
        )
        axes[2, 1].plot(
            time_axis,
            anom_att_scaled,
            label="Attention importance",
            linewidth=1.2,
            alpha=0.8,
        )
        axes[2, 1].set_title("Attention vs Reconstruction Error | Anomalous")
        axes[2, 1].set_xlabel("Timestep")
        axes[2, 1].set_ylabel("Normalized value")
        axes[2, 1].legend()
        axes[2, 1].grid(alpha=0.3)

        plt.suptitle(
            "AnoFormer Attention-based Explanation: Normal vs Anomalous ECG",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.show()

    # ──────────────────────────────────────────────────────────
    def ShowTrainingHistory(self) -> None:
        """
        Shows generator and discriminator loss curves.
        """

        History = self.Model.TrainingHistory

        if not any(History.values()):
            print("No training history available.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        epochs = range(1, len(History["GeneratorTotalLoss"]) + 1)

        axes[0].plot(epochs, History["GeneratorTotalLoss"], label="G total", linewidth=1.5)
        axes[0].plot(epochs, History["GeneratorGANLoss"], label="G GAN", linewidth=1.2, linestyle="--")
        axes[0].plot(epochs, History["GeneratorRecLoss"], label="G reconstruction", linewidth=1.2, linestyle=":")
        axes[0].set_title("Generator loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, History["DiscriminatorLoss"], label="D loss", linewidth=1.5)
        axes[1].set_title("Discriminator loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.suptitle("AnoFormer Training History", fontweight="bold")
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
