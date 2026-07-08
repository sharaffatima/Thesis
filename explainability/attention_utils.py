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
    layout = [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
    ]

    OriginalColor = "#1f4e79"       # dark blue
    ReconstructedColor = "#d95f02"  # orange

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(16, 7),
        sharex=True,
    )

    for row_idx in range(3):
        for col_idx in range(4):
            lead_name = layout[row_idx][col_idx]
            lead_idx = ECG_LEADS.index(lead_name)

            ax = axes[row_idx, col_idx]

            ax.plot(
                time_axis,
                original[:, lead_idx],
                color=OriginalColor,
                linewidth=1.1,
                label="Original" if row_idx == 0 and col_idx == 0 else None,
            )

            ax.plot(
                time_axis,
                reconstructed[:, lead_idx],
                color=ReconstructedColor,
                linestyle="--",
                linewidth=1.1,
                label="Reconstructed" if row_idx == 0 and col_idx == 0 else None,
            )

            values = np.concatenate([
                original[:, lead_idx],
                reconstructed[:, lead_idx],
            ])

            y_min = values.min()
            y_max = values.max()
            margin = (y_max - y_min) * 0.20

            if margin == 0:
                margin = 0.1

            ax.set_ylim(
                y_min - margin,
                y_max + margin,
            )

            ax.text(
                0.02,
                0.85,
                lead_name,
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
            )

            ax.grid(
                True,
                alpha=0.25,
            )

            ax.tick_params(
                axis="both",
                labelsize=8,
            )

            ax.set_yticklabels([])

    axes[0, 0].legend(
        fontsize=9,
        loc="upper right",
    )

    for ax in axes[-1, :]:
        ax.set_xlabel("Timestep")

    fig.suptitle(
        f"{model_name} {sample_type} ECG | Original vs Reconstructed | Sample {sample_index}",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])

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

