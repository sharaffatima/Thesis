import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity


class FaithfulnessEvaluation:
    """
    Faithfulness evaluation for explanation methods.
    Uses timestep-level temporal importance values.
    """

    def __init__(self, top_fraction=0.10, epsilon=1e-8):
        self.top_fraction = top_fraction
        self.epsilon = epsilon

    def mask_top_timesteps(self, ecg, temporal_importance):
        """
        Masks the top-ranked timesteps based on timestep-level importance.
        """

        sequence_length = ecg.shape[1]

        # Prepare importance vector
        importance = np.asarray(temporal_importance, dtype=float).reshape(-1)

        if len(importance) != sequence_length:
            raise ValueError(
                f"temporal_importance must have length {sequence_length}, "
                f"but got {len(importance)}."
            )

        # Select top 10% timesteps
        number_selected = max(
            1,
            int(np.ceil(sequence_length * self.top_fraction))
        )

        selected_timesteps = np.argsort(importance)[-number_selected:]
        selected_timesteps = np.sort(selected_timesteps)

        # Boolean mask for selected timesteps
        timestep_mask = np.zeros(sequence_length, dtype=bool)
        timestep_mask[selected_timesteps] = True

        # Copy ECG before perturbation
        perturbed_ecg = ecg.detach().clone()

        # Find consecutive selected timestep regions
        changes = np.diff(np.pad(timestep_mask.astype(int), (1, 1)))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        masked_ranges = []

        for start, end in zip(starts, ends):
            start = int(start)
            end = int(end)

            masked_ranges.append((start, end))

            region_length = end - start

            # Normal case: selected region is between two valid neighbors
            if start > 0 and end < sequence_length:
                left_value = ecg[:, start - 1, :]
                right_value = ecg[:, end, :]

                weights = torch.linspace(
                    0,
                    1,
                    region_length + 2,
                    device=ecg.device,
                    dtype=ecg.dtype,
                )[1:-1].view(1, region_length, 1)

                perturbed_ecg[:, start:end, :] = (
                    left_value.unsqueeze(1) * (1 - weights)
                    + right_value.unsqueeze(1) * weights
                )

            # Selected region starts at the beginning
            elif start == 0 and end < sequence_length:
                perturbed_ecg[:, start:end, :] = ecg[:, end:end + 1, :]

            # Selected region reaches the end
            elif start > 0 and end == sequence_length:
                perturbed_ecg[:, start:end, :] = ecg[:, start - 1:start, :]

            # Whole ECG selected, unlikely but safe fallback
            else:
                lead_mean = ecg.mean(dim=1, keepdim=True)
                perturbed_ecg[:, start:end, :] = lead_mean

        return perturbed_ecg, selected_timesteps, masked_ranges

    def evaluate(self, ecg, temporal_importance, score_function):
        """
        Evaluates Faithfulness for one ECG and one explanation method.
        """

        # Original anomaly score
        original_score = score_function(ecg)

        # Mask top timesteps
        perturbed_ecg, selected_timesteps, masked_ranges = self.mask_top_timesteps(
            ecg=ecg,
            temporal_importance=temporal_importance,
        )

        # Perturbed anomaly score
        perturbed_score = score_function(perturbed_ecg)

        # Faithfulness score
        score_difference = perturbed_score - original_score

        faithfulness_score = (
            abs(score_difference) / (
            abs(original_score) + self.epsilon)
        )

        return {
            "original_score": original_score,
            "perturbed_score": perturbed_score,
            "score_difference": score_difference,
            "faithfulness_score": faithfulness_score,
            "selected_timesteps": selected_timesteps,
            "masked_ranges": masked_ranges,
            "perturbed_ecg": perturbed_ecg,
        }


class StabilityEvaluation:
    """
    Stability evaluation for explanation methods.
    Uses small Gaussian noise and mean cosine similarity.
    """

    def __init__(self, noise_ratio=0.01, epsilon=1e-8):
        self.noise_ratio = noise_ratio
        self.epsilon = epsilon

    def add_gaussian_noise(self, ecg):
        """
        Add Gaussian noise independently to each ECG lead.
        The noise standard deviation is 1% of the standard deviation
        of each corresponding ECG lead by default.
        """

        if isinstance(ecg, np.ndarray):
            ecg = torch.tensor(ecg, dtype=torch.float32)
        
        if ecg.ndim == 2:
            ecg = ecg.unsqueeze(0)

        # Standard deviation per lead: shape (1, 1, 12)
        lead_std = ecg.std(dim=1, keepdim=True, unbiased=False)
        
        noise = torch.randn_like(ecg) * self.noise_ratio * lead_std
        noisy_ecg = ecg + noise

        return noisy_ecg

    def evaluate(self, original_importance, noisy_importance_values):
        """
        Calculate mean cosine similarity between the original explanation
        and the explanations of noisy ECG variants
        """
        original_importance = np.asarray(original_importance, dtype=float).reshape(1, -1)

        cosine_similarities = []

        for noisy_importance in noisy_importance_values:
            noisy_importance = np.asarray(noisy_importance, dtype=float).reshape(1, -1)
            similarity = cosine_similarity(
                original_importance,
                noisy_importance
            )[0, 0]
            cosine_similarities.append(similarity)

        cosine_similarities = np.asarray(cosine_similarities, dtype=float)
        stability_score = float(cosine_similarities.mean())

        if len(cosine_similarities) > 1:
            stability_std = float(cosine_similarities.std(ddof=1))
        else:
            stability_std = 0.0

        return {
            "stability_score": stability_score,
            "stability_std": stability_std,
            "cosine_similarities": cosine_similarities,
        }


class ReconstructionErrorAgreementEvaluation:
    """
    Reconstruction-error agreement evaluation.
    Compares timestep-level explanation importance with timestep-level reconstruction error using spearman rank correlation.
    """

    def evaluate(self, temporal_importance, reconstruction_error):
        """
        Takes temporal importance vector shape (1000,) and reconstruction error vector shape (1000,)
        and returns a dictionary containing the Spearman correlation
        """

        temporal_importance = np.asarray(
            temporal_importance,
            dtype=float,
        ).reshape(-1)

        reconstruction_error = np.asarray(
            reconstruction_error,
            dtype=float,
        ).reshape(-1)

        if temporal_importance.shape != reconstruction_error.shape:
            raise ValueError(
                f"temporal_importance and reconstruction_error must have "
                f"the same shape, but got {temporal_importance.shape} and "
                f"{reconstruction_error.shape}." 
            )
        
        correlation, _ = spearmanr(
            temporal_importance,
            reconstruction_error
        )
        return{"spearman_correlation": float(correlation) }

