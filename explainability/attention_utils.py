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
    time_axis,
    save_path=None,
):
    OriginalColor = "#1f4e79"       # dark blue
    ReconstructedColor = "#d95f02"  # orange

    num_leads = len(ECG_LEADS)

    LeadSpacing = 1.0
    SignalHeight = 0.35

    fig, ax = plt.subplots(
        figsize=(16, 10)
    )

    for lead_idx, lead_name in enumerate(ECG_LEADS):
        offset = (num_leads - 1 - lead_idx) * LeadSpacing

        values = np.concatenate([
            original[:, lead_idx],
            reconstructed[:, lead_idx],
        ])

        scale = np.percentile(
            np.abs(values),
            99,
        )

        if scale == 0:
            scale = 1.0

        original_scaled = (
            original[:, lead_idx] / scale
        ) * SignalHeight

        reconstructed_scaled = (
            reconstructed[:, lead_idx] / scale
        ) * SignalHeight

        ax.plot(
            time_axis,
            original_scaled + offset,
            color=OriginalColor,
            linewidth=1.2,
            label="Original" if lead_idx == 0 else None,
        )

        ax.plot(
            time_axis,
            reconstructed_scaled + offset,
            color=ReconstructedColor,
            linestyle="--",
            linewidth=1.2,
            label="Reconstructed" if lead_idx == 0 else None,
        )

        # horizontal separator line
        ax.axhline(
            offset - 0.5 * LeadSpacing,
            color="gray",
            linewidth=0.6,
            alpha=0.3,
        )

    ytick_positions = [
        (num_leads - 1 - i) * LeadSpacing
        for i in range(num_leads)
    ]

    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(
        ECG_LEADS,
        fontsize=12,
        fontweight="bold",
    )

    ax.set_ylim(
        -0.7,
        num_leads - 0.3,
    )

    ax.set_xlabel(
        "Timestep",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_title(
        f"{model_name} {sample_type} ECG | Original vs Reconstructed | Sample {sample_index}",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    ax.legend(
        fontsize=10,
        loc="upper right",
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
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

