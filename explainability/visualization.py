import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Helper functions for Plotting comparable Attention and SHAP visualisations
# Those are mainly for the temporal importance plots

def normalize_importance(values):
    """
    Normalize the importance values to the range [0, 1].
    This is helpful because it makes the importance values for attention and SHAP comparabel
    """
    values = np.asarray(values, dtype=np.float64)

    value_min = values.min()
    value_max = values.max()

    if np.isclose(value_min, value_max):
        return np.zeros_like(values)

    return (values - value_min) / (value_max - value_min)




ECG_LEADS = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3",
    "V4", "V5", "V6",
]


BaseCmap = plt.get_cmap("RdYlBu_r")

BrightImportanceCmap = LinearSegmentedColormap.from_list(
    "BrightImportance",
    BaseCmap(np.linspace(0.10, 0.90, 256)),
)



def plot_temporal_importance(
    values,
    title,
    ecg,
    sampling_rate=100,
    save_path=None,
):
    normalized = normalize_importance(values)

    ecg = np.asarray(ecg)

    if ecg.ndim == 3:
        ecg = ecg[0]

    if ecg.shape[1] != 12:
        raise ValueError("ECG must have shape (T, 12).")

    sequence_length = ecg.shape[0]
    duration_seconds = sequence_length / sampling_rate
    time_axis = np.arange(sequence_length) / sampling_rate

    figure, axis = plt.subplots(
        figsize=(14, 8)
    )

    # Temporal attention displayed as vertical stripes
    image = axis.imshow(
        normalized[np.newaxis, :],
        aspect="auto",
        interpolation="nearest",
        cmap=BrightImportanceCmap,
        vmin=0,
        vmax=1,
        extent=[
            0,
            duration_seconds,
            -0.5,
            11.5,
        ],
        alpha=0.70,
    )

    row_positions = np.arange(11, -1, -1)

    # Plot each lead in the same visible color
    for lead_index in range(12):
        signal = ecg[:, lead_index]

        # Center and scale only for stacked visualization
        signal = signal - np.median(signal)

        amplitude = np.percentile(
            np.abs(signal),
            99,
        )

        if amplitude > 0:
            signal = signal / amplitude

        signal = (
            signal * 0.35
            + row_positions[lead_index]
        )

        axis.plot(
            time_axis,
            signal,
            color="#17324D",
            linewidth=0.9,
        )

    axis.set_xlim(
        0,
        duration_seconds,
    )

    axis.set_ylim(
        -0.5,
        11.5,
    )

    axis.set_yticks(row_positions)
    axis.set_yticklabels(ECG_LEADS)

    axis.set_title(title)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("ECG leads")

    colorbar = figure.colorbar(
        image,
        ax=axis,
        pad=0.02,
    )

    colorbar.set_label("Normalized temporal importance")

    figure.tight_layout()

    if save_path is not None:
        figure.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()