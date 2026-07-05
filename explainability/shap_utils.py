import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from explainability.visualization import BrightCmap


def aggregate_shap_intervals(
    shap_values: np.ndarray,
    block_size: int = 10,
    absolute: bool = True,
) -> np.ndarray:
    """
    Aggregate SHAP values into temporal intervals
    """
    if shap_values.ndim != 2:
        raise ValueError(
            f"Expected (time, leads), got {shap_values.shape}"
        )

    values = np.abs(shap_values) if absolute else shap_values

    time_steps, num_leads = values.shape
    num_blocks = time_steps // block_size

    values = values[:num_blocks * block_size]

    return values.reshape(
        num_blocks,
        block_size,
        num_leads,
    ).mean(axis=1)



def plot_normal_anomalous_shap_heatmaps(
    normal_values: np.ndarray,
    anomalous_values: np.ndarray,
    lead_names: list[str],
    interval_seconds: float = 0.1,
    percentile: float = 99.0,
    save_path: Path | str | None = None,
) -> None:

    if normal_values.ndim != 2 or anomalous_values.ndim != 2:
        raise ValueError(
            "Both inputs must have shape (intervals, leads)."
        )

    if normal_values.shape != anomalous_values.shape:
        raise ValueError(
            "Normal and anomalous explanations must have the same shape."
        )

    if normal_values.shape[1] != len(lead_names):
        raise ValueError(
            "The number of leads does not match lead_names."
        )

    # Convert from (intervals, leads) to (leads, intervals).
    normal_heatmap = normal_values.T
    anomalous_heatmap = anomalous_values.T

    # One shared scale makes the two samples directly comparable.
    combined_values = np.concatenate([
        normal_heatmap.ravel(),
        anomalous_heatmap.ravel(),
    ])

    shared_vmax = np.percentile(
        combined_values,
        percentile,
    )

    if shared_vmax <= 0:
        shared_vmax = 1e-12

    duration_seconds = (
        normal_values.shape[0] * interval_seconds
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(20, 5),
        sharey=False,
        constrained_layout=True,
    )

    titles = ["Normal ECG","Anomalous ECG"]
    heatmaps = [normal_heatmap, anomalous_heatmap]
    image = None

    for axis, heatmap, title in zip(
        axes,
        heatmaps,
        titles,
    ):
        image = axis.imshow(
            heatmap,
            aspect="auto",
            interpolation="nearest",
            cmap=BrightCmap,
            vmin=0,
            vmax=shared_vmax,
            extent=[
                0,
                duration_seconds,
                len(lead_names) - 0.5,
                -0.5,
            ],
        )

        axis.set_title(title)
        axis.set_xlabel("Time (seconds)")
        axis.set_yticks(np.arange(len(lead_names)))
        axis.set_yticklabels(lead_names)

    axes[0].set_ylabel("ECG lead")

    figure.suptitle(
        "AnoFormer mean absolute SHAP importance",
        fontsize=14,
    )

    colorbar = figure.colorbar(
        image,
        ax=axes,
        location="right",
        shrink=0.95,
        pad=0.02,
    )

    colorbar.set_label("Mean absolute SHAP value")

    if save_path is not None:
        figure.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()