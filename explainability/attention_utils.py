import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ECG_LEADS = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", 
    "V4", "V5", "V6"
]

LEAD_GROUPS = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [9, 10, 11],
]

GROUP_TITLES = [
    "Leads I, II, III",
    "Leads aVR, aVL, aVF",
    "Leads V1, V2, V3",
    "Leads V4, V5, V6",
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
        time_axis):
    
    fig1, axes1 = plt.subplots(4, 2, figsize=(22, 12), sharex=True)

    for group_idx, lead_group in enumerate(LEAD_GROUPS):
        ax_normal = axes1[group_idx, 0]
        ax_anom = axes1[group_idx, 1]

        group_values = []

        for lead_idx in lead_group:
            group_values.extend([
                normal_original[:, lead_idx],
                normal_rec[:, lead_idx],
                anom_original[:, lead_idx],
                anom_rec[:, lead_idx],
            ])

        group_values = np.concatenate(group_values)

        group_low = np.percentile(group_values, 1)
        group_high = np.percentile(group_values, 99)
        group_range = group_high - group_low

        LeadOffset = max(group_range * 1.4, 1.5)

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

        ytick_labels = [
            ECG_LEADS[lead_idx]
            for lead_idx in lead_group
        ]

        y_min = -0.75 * LeadOffset
        y_max = (len(lead_group) - 1) * LeadOffset + 0.75 * LeadOffset

        for ax in [ax_normal, ax_anom]:
            ax.set_yticks(ytick_positions)
            ax.set_yticklabels(ytick_labels)
            ax.set_ylim(y_min, y_max)
            ax.grid(alpha=0.25)

            # NEW: show x-axis on every subplot
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_xlabel("Timestep")

        ax_normal.set_title(
            f"Normal ECG | {GROUP_TITLES[group_idx]}",
            fontsize=10,
        )

        ax_anom.set_title(
            f"Anomalous ECG | {GROUP_TITLES[group_idx]}",
            fontsize=10,
        )

    axes1[0, 0].legend(fontsize=8, loc="upper right")
    axes1[0, 1].legend(fontsize=8, loc="upper right")

    fig1.suptitle(
        f"{model_name} Original vs Reconstructed ECG | Normal Sample {norm_index} vs Anomalous Sample {anom_index}",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()
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

