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
    OriginalColor = "#1f4e79"       # dark blue
    ReconstructedColor = "#d95f02"  # orange

    num_leads = len(ECG_LEADS)

    fig, axes = plt.subplots(
        num_leads,
        2,
        figsize=(16, 18),
        sharex=True,
    )

    for lead_idx, lead_name in enumerate(ECG_LEADS):

        ax_normal = axes[lead_idx, 0]
        ax_anom = axes[lead_idx, 1]

        # Normal ECG
        ax_normal.plot(
            time_axis,
            normal_original[:, lead_idx],
            color=OriginalColor,
            linewidth=1.1,
            label="Original" if lead_idx == 0 else None,
        )

        ax_normal.plot(
            time_axis,
            normal_rec[:, lead_idx],
            color=ReconstructedColor,
            linestyle="--",
            linewidth=1.1,
            label="Reconstructed" if lead_idx == 0 else None,
        )

        # Anomalous ECG
        ax_anom.plot(
            time_axis,
            anom_original[:, lead_idx],
            color=OriginalColor,
            linewidth=1.1,
            label="Original" if lead_idx == 0 else None,
        )

        ax_anom.plot(
            time_axis,
            anom_rec[:, lead_idx],
            color=ReconstructedColor,
            linestyle="--",
            linewidth=1.1,
            label="Reconstructed" if lead_idx == 0 else None,
        )

        # Separate y-limits per lead and sample
        normal_values = np.concatenate([
            normal_original[:, lead_idx],
            normal_rec[:, lead_idx],
        ])

        anom_values = np.concatenate([
            anom_original[:, lead_idx],
            anom_rec[:, lead_idx],
        ])

        normal_min = normal_values.min()
        normal_max = normal_values.max()
        normal_margin = (normal_max - normal_min) * 0.15

        anom_min = anom_values.min()
        anom_max = anom_values.max()
        anom_margin = (anom_max - anom_min) * 0.15

        ax_normal.set_ylim(
            normal_min - normal_margin,
            normal_max + normal_margin,
        )

        ax_anom.set_ylim(
            anom_min - anom_margin,
            anom_max + anom_margin,
        )

        ax_normal.set_ylabel(
            lead_name,
            rotation=0,
            labelpad=18,
            fontsize=9,
        )

        for ax in [ax_normal, ax_anom]:
            ax.grid(axis="x", alpha=0.25)
            ax.tick_params(axis="y", labelleft=False)
            ax.tick_params(axis="x", labelsize=8)

    axes[0, 0].set_title(
        f"Normal ECG | Sample {norm_index}",
        fontsize=13,
        fontweight="bold",
    )

    axes[0, 1].set_title(
        f"Anomalous ECG | Sample {anom_index}",
        fontsize=13,
        fontweight="bold",
    )

    axes[0, 0].legend(
        fontsize=9,
        loc="upper right",
    )

    axes[0, 1].legend(
        fontsize=9,
        loc="upper right",
    )

    axes[-1, 0].set_xlabel("Timestep")
    axes[-1, 1].set_xlabel("Timestep")

    fig.suptitle(
        f"{model_name} Original vs Reconstructed ECG",
        fontsize=16,
        fontweight="bold",
    )

    fig.subplots_adjust(
        hspace=0.35,
        wspace=0.12,
        top=0.94,
        bottom=0.05,
        left=0.06,
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

