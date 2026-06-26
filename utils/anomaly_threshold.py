# Threshold calculation method based on Validation scores
import numpy as np
from sklearn.metrics import precision_recall_curve

# Threshold calculation method based on Validation scores


def ChooseThresholdForRecall(
    Labels: np.ndarray,
    Scores: np.ndarray,
    TargetRecall: float,
) -> dict:
    """
    Selects a threshold on the validation set.

    The threshold must achieve at least TargetRecall.
    Among all valid thresholds, the one with the highest
    validation precision is selected.
    """

    PrecisionValues, RecallValues, ThresholdValues = (
        precision_recall_curve(
            Labels,
            Scores,
        )
    )

    # The final precision and recall values do not have
    # a corresponding threshold.
    PrecisionValues = PrecisionValues[:-1]
    RecallValues = RecallValues[:-1]

    ValidIndices = np.where(
        RecallValues >= TargetRecall
    )[0]

    if len(ValidIndices) == 0:
        raise ValueError(
            f"No threshold achieved recall >= {TargetRecall:.2f}."
        )

    ValidPrecision = PrecisionValues[ValidIndices]

    HighestPrecision = ValidPrecision.max()

    BestCandidates = ValidIndices[
        np.isclose(
            ValidPrecision,
            HighestPrecision,
        )
    ]

    # In case of a tie, choose the highest threshold.
    # This usually reduces false positives while still
    # satisfying the recall requirement.
    BestIndex = BestCandidates[
        np.argmax(ThresholdValues[BestCandidates])
    ]

    return {
        "Threshold": float(ThresholdValues[BestIndex]),
        "Precision": float(PrecisionValues[BestIndex]),
        "Recall": float(RecallValues[BestIndex]),
    }