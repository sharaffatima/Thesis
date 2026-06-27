# Evaluation of Attention based Explainability and SHAP based Explainability
# Stability, Sensitivity, Plausibility

import numpy as np
import matplotlib.pyplot as plt

# Helper functions for Plotting comparable Attention and SHAP visualisations

def min_max_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    minimum = values.min()
    maximum = values.max()

    if np.isclose(maximum, minimum):
        return np.zeros_like(values)

    return (values - minimum) / (maximum - minimum)


def plot_temporal_explanation(
    importance: np.ndarray,
    title: str,
    interval_seconds: float = 0.1,
    save_path=None,
) -> None:
    """
    Plot one normalized temporal explanation.

    importance shape: (number_of_intervals,)
    """

    importance = min_max_normalize(importance)
    heatmap = importance[np.newaxis, :]  # (1, intervals)

    duration = importance.shape[0] * interval_seconds

    figure, axis = plt.subplots(figsize=(14, 2.2))

    image = axis.imshow(
        heatmap,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=1,
        extent=[0, duration, 0, 1],
    )

    axis.set_title(title)
    axis.set_xlabel("Time (seconds)")
    axis.set_yticks([0.5])
    axis.set_yticklabels(["Importance"])

    colorbar = figure.colorbar(
        image,
        ax=axis,
        pad=0.02,
    )
    colorbar.set_label("Normalized importance")

    figure.tight_layout()

    if save_path is not None:
        figure.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()