import numpy as np
import torch
import matplotlib.pyplot as plt
from .attention_utils import plot_original_vs_reconstruction, plot_attention_vs_error, BrightCmap

class TranADExplainability:
    """
    Attention-based explainability for TranAD on ECG data
    """

    def __init__(self, TranADModel):
        self.Model  = TranADModel
        self.Device = TranADModel.Device
    
    def _get_attention_matrix(
        self,
        TimeSeriesWindow: torch.Tensor,
        Component: str = "Encoder",
        LayerIndex: int = 0,
        HeadIndex: int | None = None,
    ) -> np.ndarray:

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

    # Aggregates the attention matrix into blocks for visualization
    def _aggregate_attention_matrix(
        self,
        AttentionMatrix: np.ndarray,
        BlockSize: int = 10,
    ) -> np.ndarray:

        if AttentionMatrix.ndim != 2:
            raise ValueError("AttentionMatrix must have shape (T, T).")

        if AttentionMatrix.shape[0] != AttentionMatrix.shape[1]:
            raise ValueError("AttentionMatrix must be square.")

        T = AttentionMatrix.shape[0]
        NumBlocks = T // BlockSize

        if NumBlocks == 0:
            raise ValueError("BlockSize is larger than the attention matrix.")

        TrimmedLength = NumBlocks * BlockSize
        TrimmedMatrix = AttentionMatrix[:TrimmedLength, :TrimmedLength]

        AggregatedMatrix = np.zeros((NumBlocks, NumBlocks), dtype=np.float32)

        for QueryBlockIndex in range(NumBlocks):
            for KeyBlockIndex in range(NumBlocks):
                Block = TrimmedMatrix[
                    QueryBlockIndex * BlockSize:(QueryBlockIndex + 1) * BlockSize,
                    KeyBlockIndex * BlockSize:(KeyBlockIndex + 1) * BlockSize,
                ]
                AggregatedMatrix[QueryBlockIndex, KeyBlockIndex] = Block.sum(axis=1).mean()
        return AggregatedMatrix 


    def get_attention_importance(
        self,
        TimeSeriesWindow: torch.Tensor,
        Component: str = "Encoder",
        LayerIndex: int = 0,
        HeadIndex: int | None = None,
    ) -> np.ndarray:
        """
        Returns mean attention received per timestep
        """

        AttentionMatrix = self._get_attention_matrix(
            TimeSeriesWindow=TimeSeriesWindow,
            Component=Component,
            LayerIndex=LayerIndex,
            HeadIndex=HeadIndex,
        )

        AttentionImportance = AttentionMatrix.mean(axis=0)

        return AttentionImportance


    def get_interval_attention_importance(
        self,
        TimeSeriesWindow: torch.Tensor,
        Component: str = "Encoder",
        LayerIndex: int = 0,
        HeadIndex: int | None = None,
        BlockSize: int = 10,
    ) -> np.ndarray:
        """
        Returns mean attention received per temporal interval
        """

        if BlockSize <= 0:
            raise ValueError("BlockSize must be greater than zero.")

        AttentionMatrix = self._get_attention_matrix(
            TimeSeriesWindow=TimeSeriesWindow,
            Component=Component,
            LayerIndex=LayerIndex,
            HeadIndex=HeadIndex,
        )

        AttentionImportance = AttentionMatrix.mean(axis=0)

        NumIntervals = len(AttentionImportance) // BlockSize

        if NumIntervals == 0:
            raise ValueError(
                "BlockSize is larger than the attention sequence."
            )

        TrimmedLength = NumIntervals * BlockSize

        TrimmedImportance = AttentionImportance[
            :TrimmedLength
        ]

        IntervalImportance = TrimmedImportance.reshape(
            NumIntervals,
            BlockSize,
        ).mean(axis=1)

        return IntervalImportance


    def get_reconstruction_error_over_time(
        self,
        TimeSeriesWindow: torch.Tensor,
        DatasetForInverseNorm,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        self.Model.Network.eval()
        InputTensor = TimeSeriesWindow.to(self.Device)

        with torch.no_grad():
            Rec1, Rec2 = self.Model.Network(InputTensor)

        OriginalNP = DatasetForInverseNorm.inverse_normalization(
            InputTensor[0].detach().cpu().numpy()
        )

        ReconstructedNP = DatasetForInverseNorm.inverse_normalization(
            Rec2[0].detach().cpu().numpy()
        )

        ReconstructionError = ((OriginalNP - ReconstructedNP) ** 2).mean(axis=1)

        return OriginalNP, ReconstructedNP, ReconstructionError
    

    def compare_normal_and_anomalous_attention(
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
        UseIntervalHeatmap: bool = True,
        BlockSize: int = 10,
        SamplingRate: int = 100,
    ) -> None:

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


        # Figure 1: Original vs Reconstruction
        plot_original_vs_reconstruction(
            model_name = "TranAD",
            normal_original = normal_original, 
            normal_rec = normal_rec, 
            norm_index = NormalIndex, 
            anom_original = anom_original, 
            anom_rec = anom_rec, 
            anom_index = AnomalousIndex, 
            time_axis = time_axis
        )

        # Figure 2: Attention vs Reconstruction Error
        fig2, axes2 = plt.subplots(1, 2, figsize=(20, 5), sharex=True)
        
        plot_attention_vs_error(
            axes2[0],
            normal_error,
            normal_att_imp,
            f"Normal ECG | Sample {NormalIndex}",
        )

        plot_attention_vs_error(
            axes2[1],
            anom_error,
            anom_att_imp,
            f"Anomalous ECG | Sample {AnomalousIndex}",
        )

        fig2.suptitle(
            "TranAD Temporal Attention and Reconstruction Error",
            fontsize=16,
            fontweight="bold",
        )

        fig2.subplots_adjust(wspace=0.4)
        plt.tight_layout()
        plt.show()


        # Figure 3: Attention Heatmaps
        fig3, axes3 = plt.subplots(1, 2, figsize=(18, 6))

        NormalHeatmap = normal_att
        AnomalousHeatmap = anom_att

        if UseIntervalHeatmap:
            NormalHeatmap = self._aggregate_attention_matrix(
                AttentionMatrix=normal_att,
                BlockSize=BlockSize,
            )

            AnomalousHeatmap = self._aggregate_attention_matrix(
                AttentionMatrix=anom_att,
                BlockSize=BlockSize,
            )

        shared_vmax = max(
            np.percentile(NormalHeatmap, Percentile),
            np.percentile(AnomalousHeatmap, Percentile),
        )

        im0 = axes3[0].imshow(
            NormalHeatmap,
            aspect="auto",
            cmap=BrightCmap,
            interpolation="nearest",
            vmin=0,
            vmax=shared_vmax,
        )

        im1 = axes3[1].imshow(
            AnomalousHeatmap,
            aspect="auto",
            cmap=BrightCmap,
            interpolation="nearest",
            vmin=0,
            vmax=shared_vmax,
        )

        if UseIntervalHeatmap:
            NumIntervals = NormalHeatmap.shape[0]

            TickStep = max(1, NumIntervals // 10)
            TickPositions = list(range(0, NumIntervals, TickStep))

            TickLabels = [
                f"{pos * BlockSize / SamplingRate:.1f}s"
                for pos in TickPositions
            ]

            axes3[0].set_xticks(TickPositions)
            axes3[0].set_yticks(TickPositions)
            axes3[0].set_xticklabels(TickLabels, rotation=45, ha="right")
            axes3[0].set_yticklabels(TickLabels)

            axes3[1].set_xticks(TickPositions)
            axes3[1].set_yticks(TickPositions)
            axes3[1].set_xticklabels(TickLabels, rotation=45, ha="right")
            axes3[1].set_yticklabels(TickLabels)

            x_label = f"Key interval ({BlockSize / SamplingRate:.2f}s blocks)"
            y_label = f"Query interval ({BlockSize / SamplingRate:.2f}s blocks)"
            heatmap_type = f"Interval-level attention heatmap ({BlockSize} samples)"
        else:
            x_label = "Key timestep"
            y_label = "Query timestep"
            heatmap_type = "Timestep-level attention heatmap"

        axes3[0].set_title(
            f"Normal ECG | Attention Heatmap | {Component}, Layer {LayerIndex}"
            + (f", Head {HeadIndex}" if HeadIndex is not None else ", all heads averaged")
        )
        axes3[0].set_xlabel(x_label)
        axes3[0].set_ylabel(y_label)

        axes3[1].set_title(
            f"Anomalous ECG | Attention Heatmap | {Component}, Layer {LayerIndex}"
            + (f", Head {HeadIndex}" if HeadIndex is not None else ", all heads averaged")
        )
        axes3[1].set_xlabel(x_label)
        axes3[1].set_ylabel(y_label)

        fig3.colorbar(im0, ax=axes3[0], fraction=0.046, pad=0.04)
        fig3.colorbar(im1, ax=axes3[1], fraction=0.046, pad=0.04)

        fig3.suptitle(
            f"TranAD Attention Heatmaps | Normal Sample {NormalIndex} vs Anomalous Sample {AnomalousIndex}",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.show()


    def show_training_history(self) -> None:
            
        History = self.Model.TrainingHistory
        if not History["TotalLoss"]:
            print("No training history available.")
            return

        Epochs = range(1, len(History["TotalLoss"]) + 1)

        fig, Axes = plt.subplots(1, 2, figsize=(13, 4))

        Axes[0].plot(Epochs, History["TotalLoss"], linewidth=1.5, color="steelblue", label="TranAD Loss")
        Axes[0].set_title(" Training Loss")
        Axes[0].set_xlabel("Epoch")
        Axes[0].set_ylabel("Loss")
        Axes[0].legend()
        Axes[0].grid(alpha=0.3)

        Axes[1].plot(Epochs, History["FocusScore"], linewidth=1.5, color="darkorange", label="φ (Focus Score)")
        Axes[1].set_title("Focus Score φ per Epoch")
        Axes[1].set_xlabel(" Epoch")
        Axes[1].set_ylabel("φ")
        Axes[1].legend()
        Axes[1].grid(alpha=0.3)

        plt.suptitle("TranAD – Training History", fontweight="bold")
        plt.tight_layout()
        plt.show()
