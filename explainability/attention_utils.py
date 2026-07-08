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

    ax_normal = axes[0]
    ax_anom = axes[1]

    OriginalColor = "#1f4e79"       # dark blue
    ReconstructedColor = "#d95f02"  # orange

    all_values = np.concatenate([
        normal_original.reshape(-1),
        normal_rec.reshape(-1),
        anom_original.reshape(-1),
        anom_rec.reshape(-1),
    ])

    value_low = np.percentile(all_values, 1)
    value_high = np.percentile(all_values, 99)
    value_range = value_high - value_low

    # Bigger distance between leads
    LeadOffset = max(value_range * 1.9, 2.0)

    num_leads = len(ECG_LEADS)

    for lead_position, lead_idx in enumerate(range(num_leads)):
        offset = (num_leads - 1 - lead_position) * LeadOffset

        ax_normal.plot(
            time_axis,
            normal_original[:, lead_idx] + offset,
            color=OriginalColor,
            linewidth=1.1,
            label="Original" if lead_position == 0 else None,
        )

        ax_normal.plot(
            time_axis,
            normal_rec[:, lead_idx] + offset,
            color=ReconstructedColor,
            linestyle="--",
            linewidth=1.1,
            label="Reconstructed" if lead_position == 0 else None,
        )

        ax_anom.plot(
            time_axis,
            anom_original[:, lead_idx] + offset,
            color=OriginalColor,
            linewidth=1.1,
            label="Original" if lead_position == 0 else None,
        )

        ax_anom.plot(
            time_axis,
            anom_rec[:, lead_idx] + offset,
            color=ReconstructedColor,
            linestyle="--",
            linewidth=1.1,
            label="Reconstructed" if lead_position == 0 else None,
        )

    ytick_positions = [
        (num_leads - 1 - i) * LeadOffset
        for i in range(num_leads)
    ]

    y_min = -1.0 * LeadOffset
    y_max = (num_leads - 1) * LeadOffset + 1.0 * LeadOffset

    for ax in [ax_normal, ax_anom]:
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ECG_LEADS)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Lead")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(fontsize=9, loc="upper right")

    ax_normal.set_title(
        f"(a) Normal ECG | Sample {norm_index}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax_anom.set_title(
        f"(b) Anomalous ECG | Sample {anom_index}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    ax_anom.set_xlabel("Timestep")

    fig.suptitle(
        f"{model_name} Original vs Reconstructed ECG",
        fontsize=16,
        fontweight="bold",
    )

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=400,
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

