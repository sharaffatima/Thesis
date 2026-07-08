import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ECG_LEADS = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", 
    "V4", "V5", "V6"
]

BaseCmap = plt.get_cmap("RdYlBu_r")

BrightCmap = LinearSegmentedColormap.from_list(
    "BrightAttention",
    BaseCmap(
        np.linspace(
            0.10,
            0.90,
            256,
        )
    ),
)


def plot_original_vs_reconstruction(
    model_name,
    sample_type,
    original,
    reconstructed,
    sample_index,
    sampling_rate=100,
    save_path=None,
):
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)

    if original.ndim == 3:
        original = original[0]

    if reconstructed.ndim == 3:
        reconstructed = reconstructed[0]

    if original.shape[1] != 12:
        raise ValueError("Original ECG must have shape (T, 12).")

    if reconstructed.shape[1] != 12:
        raise ValueError("Reconstructed ECG must have shape (T, 12).")

    sequence_length = original.shape[0]
    duration_seconds = sequence_length / sampling_rate
    time_axis = np.arange(sequence_length) / sampling_rate

    OriginalColor = "#12263A"       # dark navy
    ReconstructedColor = "#D95F02"  # orange

    figure, axis = plt.subplots(
        figsize=(14, 8)
    )

    row_positions = np.arange(11, -1, -1)

    for lead_index in range(12):
        original_signal = original[:, lead_index]
        reconstructed_signal = reconstructed[:, lead_index]

        # Same centering and scaling for original + reconstruction
        combined = np.concatenate([
            original_signal,
            reconstructed_signal,
        ])

        center = np.median(combined)

        original_signal = original_signal - center
        reconstructed_signal = reconstructed_signal - center

        combined_centered = combined - center

        amplitude = np.percentile(
            np.abs(combined_centered),
            99,
        )

        if amplitude > 0:
            original_signal = original_signal / amplitude
            reconstructed_signal = reconstructed_signal / amplitude

        original_signal = (
            original_signal * 0.35
            + row_positions[lead_index]
        )

        reconstructed_signal = (
            reconstructed_signal * 0.35
            + row_positions[lead_index]
        )

        axis.plot(
            time_axis,
            original_signal,
            color=OriginalColor,
            linewidth=0.9,
            label="Original" if lead_index == 0 else None,
        )

        axis.plot(
            time_axis,
            reconstructed_signal,
            color=ReconstructedColor,
            linestyle="--",
            linewidth=0.9,
            label="Reconstructed" if lead_index == 0 else None,
        )

    axis.set_xlim(0, duration_seconds)
    axis.set_ylim(-0.5, 11.5)

    axis.set_yticks(row_positions)
    axis.set_yticklabels(ECG_LEADS)

    axis.set_title(
        f"{model_name} {sample_type} ECG | Original vs Reconstructed | Sample {sample_index}",
        fontsize=14,
        fontweight="bold",
    )

    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("ECG leads")

    axis.grid(
        axis="x",
        alpha=0.25,
    )

    axis.legend(
        fontsize=9,
        loc="upper right",
    )

    figure.tight_layout()

    if save_path is not None:
        figure.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_attention_vs_error(ax, error, attention, title, time_axis):
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

