import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Optional, List
from models.anoformer import AnoFormerMainModel
from models.tranad import TranADMainModel
from utils.dataset import PTBXLTranADDataset



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



        # ============================================================
        # Figure 1: Original vs Reconstruction for all 12 leads
        # ============================================================

        fig1, axes1 = plt.subplots(4, 2, figsize=(22, 12), sharex=True)

        ECG_LEADS = [
            "I", "II", "III",
            "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6"
        ]

        LeadGroups = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
        ]

        GroupTitles = [
            "Leads I, II, III",
            "Leads aVR, aVL, aVF",
            "Leads V1, V2, V3",
            "Leads V4, V5, V6",
        ]

        LeadOffset = 2.5

        for group_idx, lead_group in enumerate(LeadGroups):
            ax_normal = axes1[group_idx, 0]
            ax_anom = axes1[group_idx, 1]

            for local_idx, lead_idx in enumerate(lead_group):
                offset = (len(lead_group) - 1 - local_idx) * LeadOffset

                ax_normal.plot(
                    time_axis,
                    normal_original[:, lead_idx] + offset,
                    linewidth=0.8,
                    label="Original" if group_idx == 0 and local_idx == 0 else None,
                )
                ax_normal.plot(
                    time_axis,
                    normal_rec[:, lead_idx] + offset,
                    linestyle="--",
                    linewidth=0.8,
                    label="Reconstructed" if group_idx == 0 and local_idx == 0 else None,
                )

                ax_anom.plot(
                    time_axis,
                    anom_original[:, lead_idx] + offset,
                    linewidth=0.8,
                    label="Original" if group_idx == 0 and local_idx == 0 else None,
                )
                ax_anom.plot(
                    time_axis,
                    anom_rec[:, lead_idx] + offset,
                    linestyle="--",
                    linewidth=0.8,
                    label="Reconstructed" if group_idx == 0 and local_idx == 0 else None,
                )

            ytick_positions = [
                (len(lead_group) - 1 - i) * LeadOffset
                for i in range(len(lead_group))
            ]

            ytick_labels = [ECG_LEADS[lead_idx] for lead_idx in lead_group]

            ax_normal.set_yticks(ytick_positions)
            ax_normal.set_yticklabels(ytick_labels)
            ax_normal.set_title(f"Normal ECG | {GroupTitles[group_idx]}", fontsize=10)
            ax_normal.grid(alpha=0.25)

            ax_anom.set_yticks(ytick_positions)
            ax_anom.set_yticklabels(ytick_labels)
            ax_anom.set_title(f"Anomalous ECG | {GroupTitles[group_idx]}", fontsize=10)
            ax_anom.grid(alpha=0.25)

        axes1[0, 0].legend(fontsize=8, loc="upper right")
        axes1[0, 1].legend(fontsize=8, loc="upper right")

        axes1[-1, 0].set_xlabel("Timestep")
        axes1[-1, 1].set_xlabel("Timestep")

        fig1.suptitle(
            f"Original vs Reconstructed ECG | Normal Sample {NormalIndex} vs Anomalous Sample {AnomalousIndex}",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.show()



        # ============================================================
        # Figure 2: Attention Heatmaps
        # ============================================================

        fig2, axes2 = plt.subplots(1, 2, figsize=(18, 6))

        shared_vmax = max(
            np.percentile(normal_att, Percentile),
            np.percentile(anom_att, Percentile),
        )

        if HeadIndex is None:
            attention_title = f"Block {BlockIndex}, all heads averaged"
        else:
            attention_title = f"Block {BlockIndex}, Head {HeadIndex}"

        im0 = axes2[0].imshow(
            normal_att,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=shared_vmax,
        )
        axes2[0].set_title(f"Normal ECG | Attention Heatmap | {attention_title}")
        axes2[0].set_xlabel("Key timestep")
        axes2[0].set_ylabel("Query timestep")
        fig2.colorbar(im0, ax=axes2[0], fraction=0.046, pad=0.04)

        im1 = axes2[1].imshow(
            anom_att,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=shared_vmax,
        )
        axes2[1].set_title(f"Anomalous ECG | Attention Heatmap | {attention_title}")
        axes2[1].set_xlabel("Key timestep")
        axes2[1].set_ylabel("Query timestep")
        fig2.colorbar(im1, ax=axes2[1], fraction=0.046, pad=0.04)

        fig2.suptitle(
            f"Attention Heatmaps | Normal Sample {NormalIndex} vs Anomalous Sample {AnomalousIndex}",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.show()


        # ============================================================
        # Figure 3: Attention vs Reconstruction Error
        # Real values, separate y-axes
        # ============================================================

        fig3, axes3 = plt.subplots(
            1, 2,
            figsize=(20, 5),
            sharex=True,
        )

        def plot_attention_vs_error(ax, error, attention, title):
            ax_attention = ax.twinx()

            line1 = ax.plot(
                time_axis,
                error,
                color="tab:orange",
                label="Reconstruction error",
                linewidth=1.5,
            )

            line2 = ax_attention.plot(
                time_axis,
                attention,
                color="tab:blue",
                label="Attention importance",
                linewidth=1.5,
                linestyle="--",
            )

            ax.set_title(title)
            ax.set_xlabel("Timestep")
            ax.set_ylabel(
                "Reconstruction error (MSE across 12 leads)",
                color="tab:orange",
            )
            ax_attention.set_ylabel(
                "Mean attention received",
                color="tab:blue",
            )
            ax.tick_params(axis="y", colors="tab:orange")
            ax_attention.tick_params(axis="y", colors="tab:blue")

            ax.grid(alpha=0.3)

            lines = line1 + line2
            labels = [line.get_label() for line in lines]

            ax.legend(lines, labels, loc="upper right")


        plot_attention_vs_error(
            axes3[0],
            normal_error,
            normal_att_imp,
            f"Normal ECG | Sample {NormalIndex}"
        )

        plot_attention_vs_error(
            axes3[1],
            anom_error,
            anom_att_imp,
            f"Anomalous ECG | Sample {AnomalousIndex}"
        )

        fig3.suptitle(
            "Temporal Attention and Reconstruction Error",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()
        fig3.subplots_adjust(wspace=0.35)
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
# TranADExplainability Attention-Explainability
# ============================================================

class TranADExplainability:
    """
    Erklärt TranAD-Anomalieentscheidungen mit sechs Visualisierungsmethoden.
    / Explains TranAD anomaly decisions with six visualization methods.
    """

    def __init__(self, MainModel: TranADMainModel):
        self.Model  = MainModel           # Referenz auf Hauptmodell / Reference to main model
        self.Device = MainModel.Device    # Gerät vom Modell übernehmen / Inherit device from model

    
    def _get_attention_matrix(
        self,
        TimeSeriesWindow: torch.Tensor,
        Component: str = "Encoder",
        LayerIndex: int = 0,
        HeadIndex: int | None = None,
    ) -> np.ndarray:
        """
        Returns attention matrix of shape (T, T).

        If HeadIndex is None, attention is averaged across all heads.
        """

        self.Model.Network.eval()
        InputTensor = TimeSeriesWindow.to(self.Device)

        with torch.no_grad():
            self.Model.Network(InputTensor)

        if Component == "Encoder":
            LayerIdx = min(
                LayerIndex,
                len(self.Model.Network.EncoderStack.EncoderLayerList) - 1,
            )
            AttModule = self.Model.Network.EncoderStack.EncoderLayerList[
                LayerIdx
            ].SelfAttentionModule

        elif Component == "Decoder1":
            LayerIdx = min(
                LayerIndex,
                len(self.Model.Network.DecoderStack1.DecoderLayerList) - 1,
            )
            AttModule = self.Model.Network.DecoderStack1.DecoderLayerList[
                LayerIdx
            ].CrossAttentionModule

        else:
            LayerIdx = min(
                LayerIndex,
                len(self.Model.Network.DecoderStack2.DecoderLayerList) - 1,
            )
            AttModule = self.Model.Network.DecoderStack2.DecoderLayerList[
                LayerIdx
            ].CrossAttentionModule

        StoredWeights = AttModule.StoredAttentionWeights

        if StoredWeights is None:
            raise ValueError("No stored attention weights available.")

        if HeadIndex is None:
            AttMatrix = StoredWeights[0].mean(dim=0).detach().cpu().numpy()
        else:
            AttMatrix = StoredWeights[0, HeadIndex].detach().cpu().numpy()

        return AttMatrix
    
    

    def GetTranADReconstructionErrorOverTime(
        self,
        TimeSeriesWindow: torch.Tensor,
        DatasetForInverseNorm,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            Original data: (T, F)
            Phase-2 reconstructed data: (T, F)
            Reconstruction error over time: (T,)
        """

        self.Model.Network.eval()
        InputTensor = TimeSeriesWindow.to(self.Device)

        with torch.no_grad():
            Rec1, Rec2 = self.Model.Network(InputTensor)

        OriginalNP = DatasetForInverseNorm.InverseNormalization(
            InputTensor[0].detach().cpu().numpy()
        )

        ReconstructedNP = DatasetForInverseNorm.InverseNormalization(
            Rec2[0].detach().cpu().numpy()
        )

        ReconstructionError = ((OriginalNP - ReconstructedNP) ** 2).mean(axis=1)

        return OriginalNP, ReconstructedNP, ReconstructionError

    def CompareNormalAndAnomalousAttention(
        self,
        NormalWindow: torch.Tensor,
        AnomalousWindow: torch.Tensor,
        DatasetForInverseNorm,
        NormalIndex: int,
        AnomalousIndex: int,
        Component: str = "Encoder",
        LayerIndex: int = 0,
        HeadIndex: int | None = None,
        Percentile: float = 99.0,
    ) -> None:

        ECG_LEADS = [
            "I", "II", "III",
            "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6"
        ]

        LeadGroups = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
        ]

        GroupTitles = [
            "Leads I, II, III",
            "Leads aVR, aVL, aVF",
            "Leads V1, V2, V3",
            "Leads V4, V5, V6",
        ]

        def compute_all(Window):
            OriginalNP, ReconstructedNP, ReconstructionError = (
                self.GetTranADReconstructionErrorOverTime(
                    TimeSeriesWindow=Window,
                    DatasetForInverseNorm=DatasetForInverseNorm,
                )
            )

            AttMatrix = self._get_attention_matrix(
                TimeSeriesWindow=Window,
                Component=Component,
                LayerIndex=LayerIndex,
                HeadIndex=HeadIndex,
            )

            AttentionImportance = AttMatrix.mean(axis=0)

            return OriginalNP, ReconstructedNP, ReconstructionError, AttMatrix, AttentionImportance

        normal_original, normal_rec, normal_error, normal_att, normal_att_imp = compute_all(NormalWindow)
        anom_original, anom_rec, anom_error, anom_att, anom_att_imp = compute_all(AnomalousWindow)

        time_axis = np.arange(normal_original.shape[0])

        if HeadIndex is None:
            attention_title = f"{Component}, Layer {LayerIndex}, all heads averaged"
        else:
            attention_title = f"{Component}, Layer {LayerIndex}, Head {HeadIndex}"

        # ============================================================
        # Figure 1: Original vs Reconstruction
        # ============================================================

        fig1, axes1 = plt.subplots(4, 2, figsize=(22, 12), sharex=True)

        LeadOffset = 2.5

        for group_idx, lead_group in enumerate(LeadGroups):
            ax_normal = axes1[group_idx, 0]
            ax_anom = axes1[group_idx, 1]

            for local_idx, lead_idx in enumerate(lead_group):
                offset = (len(lead_group) - 1 - local_idx) * LeadOffset

                ax_normal.plot(
                    time_axis,
                    normal_original[:, lead_idx] + offset,
                    linewidth=0.8,
                    label="Original" if group_idx == 0 and local_idx == 0 else None,
                )
                ax_normal.plot(
                    time_axis,
                    normal_rec[:, lead_idx] + offset,
                    linestyle="--",
                    linewidth=0.8,
                    label="Reconstructed" if group_idx == 0 and local_idx == 0 else None,
                )

                ax_anom.plot(
                    time_axis,
                    anom_original[:, lead_idx] + offset,
                    linewidth=0.8,
                    label="Original" if group_idx == 0 and local_idx == 0 else None,
                )
                ax_anom.plot(
                    time_axis,
                    anom_rec[:, lead_idx] + offset,
                    linestyle="--",
                    linewidth=0.8,
                    label="Reconstructed" if group_idx == 0 and local_idx == 0 else None,
                )

            ytick_positions = [
                (len(lead_group) - 1 - i) * LeadOffset
                for i in range(len(lead_group))
            ]

            ytick_labels = [ECG_LEADS[lead_idx] for lead_idx in lead_group]

            ax_normal.set_yticks(ytick_positions)
            ax_normal.set_yticklabels(ytick_labels)
            ax_normal.set_title(f"Normal ECG | {GroupTitles[group_idx]}", fontsize=10)
            ax_normal.grid(alpha=0.25)

            ax_anom.set_yticks(ytick_positions)
            ax_anom.set_yticklabels(ytick_labels)
            ax_anom.set_title(f"Anomalous ECG | {GroupTitles[group_idx]}", fontsize=10)
            ax_anom.grid(alpha=0.25)

        axes1[0, 0].legend(fontsize=8, loc="upper right")
        axes1[0, 1].legend(fontsize=8, loc="upper right")

        axes1[-1, 0].set_xlabel("Timestep")
        axes1[-1, 1].set_xlabel("Timestep")

        fig1.suptitle(
            f"TranAD Original vs Reconstructed ECG | Normal Sample {NormalIndex} vs Anomalous Sample {AnomalousIndex}",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.show()



        # ============================================================
        # Figure 2: Attention Heatmaps
        # ============================================================

        fig2, axes2 = plt.subplots(1, 2, figsize=(18, 6))

        shared_vmax = max(
            np.percentile(normal_att, Percentile),
            np.percentile(anom_att, Percentile),
        )

        im0 = axes2[0].imshow(
            normal_att,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=shared_vmax,
        )
        axes2[0].set_title(f"Normal ECG | Attention Heatmap | {attention_title}")
        axes2[0].set_xlabel("Key timestep")
        axes2[0].set_ylabel("Query timestep")
        fig2.colorbar(im0, ax=axes2[0], fraction=0.046, pad=0.04)

        im1 = axes2[1].imshow(
            anom_att,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=shared_vmax,
        )
        axes2[1].set_title(f"Anomalous ECG | Attention Heatmap | {attention_title}")
        axes2[1].set_xlabel("Key timestep")
        axes2[1].set_ylabel("Query timestep")
        fig2.colorbar(im1, ax=axes2[1], fraction=0.046, pad=0.04)

        fig2.suptitle(
            f"TranAD Attention Heatmaps | Normal Sample {NormalIndex} vs Anomalous Sample {AnomalousIndex}",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.show()




        # ============================================================
        # Figure 3: Attention vs Reconstruction Error
        # ============================================================

        fig3, axes3 = plt.subplots(1, 2, figsize=(20, 5), sharex=True)

        def plot_attention_vs_error(ax, error, attention, title):
            ax_attention = ax.twinx()

            line1 = ax.plot(
                time_axis,
                error,
                color="tab:orange",
                label="Reconstruction error",
                linewidth=1.5,
            )

            line2 = ax_attention.plot(
                time_axis,
                attention,
                color="tab:blue",
                label="Attention importance",
                linewidth=1.5,
                linestyle="--",
            )

            ax.set_title(title)
            ax.set_xlabel("Timestep")
            ax.set_ylabel(
                "Reconstruction error (MSE across 12 leads)",
                color="tab:orange",
            )
            ax_attention.set_ylabel(
                "Mean attention received",
                color="tab:blue",
            )

            ax.tick_params(axis="y", colors="tab:orange")
            ax_attention.tick_params(axis="y", colors="tab:blue")

            ax.grid(alpha=0.3)

            lines = line1 + line2
            labels = [line.get_label() for line in lines]
            ax.legend(lines, labels, loc="upper right")

        plot_attention_vs_error(
            axes3[0],
            normal_error,
            normal_att_imp,
            f"Normal ECG | Sample {NormalIndex}",
        )

        plot_attention_vs_error(
            axes3[1],
            anom_error,
            anom_att_imp,
            f"Anomalous ECG | Sample {AnomalousIndex}",
        )

        fig3.suptitle(
            "TranAD Temporal Attention and Reconstruction Error",
            fontsize=16,
            fontweight="bold",
        )

        fig3.subplots_adjust(wspace=0.4)
        plt.tight_layout()
        plt.show()
    