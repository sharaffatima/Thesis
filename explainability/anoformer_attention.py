import numpy as np
import torch
import matplotlib.pyplot as plt
from .attention_utils import plot_original_vs_reconstruction, plot_attention_vs_error, BrightCmap

class AnoFormerExplainability:
    """
    Attention-based explainability for AnoFormer on ECG data
    """

    def __init__(self, AnoFormerModel):
        self.Model = AnoFormerModel
        self.Device = AnoFormerModel.Device

    def _get_attention_matrix(
        self,
        TimeSeriesWindow: torch.Tensor,
        BlockIndex: int = 0,
        HeadIndex: int | None = None,
    ) -> np.ndarray:

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

        if HeadIndex is None:
            AttMatrix = AttWeights[0].mean(dim=0).detach().cpu().numpy()
        else:
            AttMatrix = AttWeights[0, HeadIndex].detach().cpu().numpy()

        return AttMatrix

     # this is needed because sometimes we need to aggregate the attention matrix into blocks for visualization
    def _aggregate_attention_matrix(
        self,
        AttentionMatrix: np.ndarray,
        BlockSize: int = 10,
    ) -> np.ndarray:

        if AttentionMatrix.ndim != 2:
            raise ValueError("AttentionMatrix must have shape (T, T).")

        T = AttentionMatrix.shape[0]

        if AttentionMatrix.shape[0] != AttentionMatrix.shape[1]:
            raise ValueError("AttentionMatrix must be square.")

        NumBlocks = T // BlockSize

        if NumBlocks == 0:
            raise ValueError("BlockSize is larger than the attention matrix.")

        TrimmedLength = NumBlocks * BlockSize

        TrimmedMatrix = AttentionMatrix[:TrimmedLength, :TrimmedLength]

        AggregatedMatrix = np.zeros(
            (NumBlocks, NumBlocks),
            dtype=np.float32,
        )

        for QueryBlockIndex in range(NumBlocks):
            for KeyBlockIndex in range(NumBlocks):
                Block = TrimmedMatrix[
                    QueryBlockIndex * BlockSize:(QueryBlockIndex + 1) * BlockSize,
                    KeyBlockIndex * BlockSize:(KeyBlockIndex + 1) * BlockSize,
                ]

                # For each query timestep, sum attention going to this key interval.
                # Then average across query timesteps.
                AggregatedMatrix[QueryBlockIndex, KeyBlockIndex] = Block.sum(axis=1).mean()
        return AggregatedMatrix
    

    def get_attention_importance(
        self,
        TimeSeriesWindow: torch.Tensor,
        BlockIndex: int = 0,
        HeadIndex: int | None = None,
    ) -> np.ndarray:
        """
        Returns mean attention received per timestep.
        """

        AttMatrix = self._get_attention_matrix(
            TimeSeriesWindow=TimeSeriesWindow,
            BlockIndex=BlockIndex,
            HeadIndex=HeadIndex,
        )
        AttentionImportance = AttMatrix.mean(axis=0)
        return AttentionImportance

    def get_interval_attention_importance(
        self,
        TimeSeriesWindow: torch.Tensor,
        BlockIndex: int = 0,
        HeadIndex: int | None = None,
        BlockSize: int = 10,
    ) -> np.ndarray:
        """
        Returns mean attention received per temporal interval
        """
        AttentionImportance = self.get_attention_importance(
            TimeSeriesWindow=TimeSeriesWindow,
            BlockIndex=BlockIndex,
            HeadIndex=HeadIndex,
        )

        NumIntervals = len(AttentionImportance) // BlockSize
        TrimmedLength = NumIntervals * BlockSize

        TrimmedImportance = AttentionImportance[:TrimmedLength]

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

        self.Model.Generator.eval()
        Input = TimeSeriesWindow.to(self.Device)

        with torch.no_grad():
            Masked, _ = self.Model.Masking(Input, None)
            Reconstructed = self.Model.Generator(Masked)

        OriginalNP = DatasetForInverseNorm.inverse_normalization(
            Input[0].detach().cpu().numpy()
        )

        ReconstructedNP = DatasetForInverseNorm.inverse_normalization(
            Reconstructed[0].detach().cpu().numpy()
        )

        # Mean squared error across all 12 leads per timestep
        ReconstructionError = ((OriginalNP - ReconstructedNP) ** 2).mean(axis=1)

        return OriginalNP, ReconstructedNP, ReconstructionError


    # Plots Reconstrucion, Attention heatmaps and reconstruction vs. attention for a normal and anomalous sample
    def compare_normal_and_anomalous_attention(
        self,
        NormalWindow: torch.Tensor,
        AnomalousWindow: torch.Tensor,
        DatasetForInverseNorm,
        NormalIndex: int,
        AnomalousIndex: int,
        BlockIndex: int = 0,
        HeadIndex: int | None = None,
        Percentile: float = 99.0,
        UseIntervalHeatmap: bool = True,
        BlockSize: int = 10,
        SamplingRate: int = 100,
    ) -> None:

        def compute_all(Window):
            OriginalNP, ReconstructedNP, ReconstructionError = (
                self.get_reconstruction_error_over_time(
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


        #Figure 1: Original vs Reconstruction
        plot_original_vs_reconstruction(
            model_name="AnoFormer",
            normal_original=normal_original,
            normal_rec=normal_rec,
            norm_index=NormalIndex,
            anom_original=anom_original,
            anom_rec=anom_rec,
            anom_index=AnomalousIndex,
            time_axis=time_axis,
            save_path="anoformer_original_vs_reconstruction.png",
        )


        # Figure 2: Attention vs Reconstruction Error
        fig2, axes2 = plt.subplots(1, 2, figsize=(20, 5), sharex=True)

        plot_attention_vs_error(
            axes2[0],
            normal_error,
            normal_att_imp,
            f"Normal ECG | Sample {NormalIndex}",
            time_axis=time_axis
        )

        plot_attention_vs_error(
            axes2[1],
            anom_error,
            anom_att_imp,
            f"Anomalous ECG | Sample {AnomalousIndex}",
            time_axis=time_axis
        )

        fig2.suptitle(
            "Temporal Attention and Reconstruction Error",
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

        axes3[0].set_title(f"Normal ECG | {heatmap_type}")
        axes3[0].set_xlabel(x_label)
        axes3[0].set_ylabel(y_label)

        axes3[1].set_title(f"Anomalous ECG | {heatmap_type}")
        axes3[1].set_xlabel(x_label)
        axes3[1].set_ylabel(y_label)

        fig3.colorbar(im0, ax=axes3[0], fraction=0.046, pad=0.04)
        fig3.colorbar(im1, ax=axes3[1], fraction=0.046, pad=0.04)

        fig3.suptitle(
            f"AnoFormer Attention Heatmaps | Normal Sample {NormalIndex} vs Anomalous Sample {AnomalousIndex}",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.show()


    def show_training_history(self) -> None:

        History = self.Model.TrainingHistory

        if not any(History.values()):
            print("No training history available.")
            return

        epochs = np.arange(
            1,
            len(History["GeneratorTotalLoss"]) + 1,
        )

        standard_epochs = epochs % 2 == 1
        guided_epochs = epochs % 2 == 0

        generator_total = np.asarray(
            History["GeneratorTotalLoss"]
        )

        generator_rec = np.asarray(
            History["GeneratorRecLoss"]
        )

        discriminator_loss = np.asarray(
            History["DiscriminatorLoss"]
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(16, 4),
        )

        axes[0].plot(
            epochs[standard_epochs],
            generator_total[standard_epochs],
            marker="o",
            label="Standard masking",
        )

        axes[0].plot(
            epochs[guided_epochs],
            generator_total[guided_epochs],
            marker="o",
            label="Error-guided masking",
        )

        axes[0].set_title("Generator total loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(alpha=0.3)


        axes[1].plot(
            epochs[standard_epochs],
            generator_rec[standard_epochs],
            marker="o",
            label="Standard masking",
        )

        axes[1].plot(
            epochs[guided_epochs],
            generator_rec[guided_epochs],
            marker="o",
            label="Error-guided masking",
        )

        axes[1].set_title("Reconstruction loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(alpha=0.3)


        axes[2].plot(
            epochs[standard_epochs],
            discriminator_loss[standard_epochs],
            marker="o",
            label="Standard masking",
        )

        axes[2].plot(
            epochs[guided_epochs],
            discriminator_loss[guided_epochs],
            marker="o",
            label="Error-guided masking",
        )

        axes[2].set_title("Discriminator loss")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.suptitle(
            "AnoFormer Training History by Masking Strategy",
            fontweight="bold",
        )

        plt.tight_layout()
        plt.show()