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
    normal_original,
    normal_rec,
    norm_index,
    anom_original,
    anom_rec,
    anom_index,
    time_axis,
    save_path=None,
):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 14),
        sharex=True,
    )

    OriginalColor = "#1f4e79"       # dark blue
    ReconstructedColor = "#d95f02"  # orange

    LeadSpacing = 1.0
    SignalHeight = 0.38

    num_leads = len(ECG_LEADS)

    def plot_sample(ax, original, reconstructed, title):
        for lead_idx in range(num_leads):
            offset = (num_leads - 1 - lead_idx) * LeadSpacing

            values = np.concatenate([
                original[:, lead_idx],
                reconstructed[:, lead_idx],
            ])

            scale = np.max(np.abs(values))

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
                linewidth=1.1,
                label="Original" if lead_idx == 0 else None,
            )

            ax.plot(
                time_axis,
                reconstructed_scaled + offset,
                color=ReconstructedColor,
                linestyle="--",
                linewidth=1.1,
                label="Reconstructed" if lead_idx == 0 else None,
            )

        ytick_positions = [
            (num_leads - 1 - i) * LeadSpacing
            for i in range(num_leads)
        ]

        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ECG_LEADS)
        ax.set_ylim(-0.7, num_leads - 0.3)
        ax.set_ylabel("Lead")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.grid(axis="x", alpha=0.25)
        ax.legend(fontsize=9, loc="upper right")

    plot_sample(
        axes[0],
        normal_original,
        normal_rec,
        f"(a) Normal ECG | Sample {norm_index}",
    )

    plot_sample(
        axes[1],
        anom_original,
        anom_rec,
        f"(b) Anomalous ECG | Sample {anom_index}",
    )

    axes[1].set_xlabel("Timestep")

    fig.suptitle(
        f"{model_name} Original vs Reconstructed ECG",
        fontsize=16,
        fontweight="bold",
    )

    fig.subplots_adjust(
        hspace=0.32,
        top=0.92,
        bottom=0.07,
        left=0.07,
        right=0.98,
    )

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

