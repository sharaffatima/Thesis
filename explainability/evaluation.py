# Evaluation of Attention based Explainability and SHAP based Explainability
# Stability, Sensitivity, Plausibility

import numpy as np
import matplotlib.pyplot as plt

# Helper functions for Plotting comparable Attention and SHAP visualisations

def normalize_importance(values):
    values = np.asarray(values, dtype=np.float64)

    value_min = values.min()
    value_max = values.max()

    if np.isclose(value_min, value_max):
        return np.zeros_like(values)

    return (values - value_min) / (value_max - value_min)


def plot_temporal_importance(
    values,
    title,
    save_path=None,
):
    normalized = normalize_importance(values)

    figure, axis = plt.subplots(figsize=(14, 2.2))

    image = axis.imshow(
        normalized[np.newaxis, :],
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=1,
        extent=[0, 10, 0, 1],
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