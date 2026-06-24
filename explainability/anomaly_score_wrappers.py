import torch
import torch.nn as nn

# ============================================================
# AnoFormer Anomaly Score Wrapper for SHAP
# ============================================================
class AnoFormerAnomalyScoreWrapper(nn.Module):
    """
    Wraps a trained AnoFormer generator and returns one reconstruction-based
    anomaly score per ECG.

    Input:
        Tensor with shape (batch, time, features)

    Output:
        Tensor with shape (batch, 1)
    """   
    def __init__(
        self, 
        anoformer_model, 
        window_size:int,
        masking_rate: float | None = None,
        num_masks:int = 1,
        seed:int = 42,
    ):
        super().__init__()

        if num_masks < 1:
            raise ValueError("num_masks must be >= 1")
        
        self.generator = anoformer_model.Generator  #only generator because the discriminator is used only during raining not during anomaly detection

        if masking_rate is None:
            masking_rate = anoformer_model.Masking.RandomMaskingRate

        self.window_size = window_size
        self.masking_rate = masking_rate
        self.num_masks = num_masks

        self.generator.eval()  # Set the generator to evaluation mode

        # We need gradients with respect to the ECG input, not the weights.
        for parameter in self.generator.parameters():
            parameter.requires_grad_(False)

        # Generate the masks once, making all later forward passes deterministic.
        random_generator = torch.Generator()
        random_generator.manual_seed(seed)

        fixed_masks = (
            torch.rand(
                num_masks,
                window_size,
                generator=random_generator,
            )
            > masking_rate
        ).float()

        # Shape: (num_masks, time, 1)
        # The final dimension allows the mask to be broadcast over all 12 leads.
        fixed_masks = fixed_masks.unsqueeze(-1)             

        # register_buffer ensures masks move to GPU together with the wrapper,
        # but they are not treated as trainable parameters.
        self.register_buffer("fixed_masks", fixed_masks)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
                    raise ValueError(
                        "Expected input shape (batch, time, features), "
                        f"but received {tuple(x.shape)}."
                    )

        if x.shape[1] != self.window_size:
            raise ValueError(
                f"Expected {self.window_size} time points, "
                f"but received {x.shape[1]}."
            )

        scores_for_masks = []

        for mask in self.fixed_masks:
            # mask shape: (time, 1)
            # after unsqueeze: (1, time, 1)
            # broadcasting applies it to every ECG and every lead.
            masked_x = x * mask.unsqueeze(0)

            reconstruction = self.generator(masked_x)

            # Mean squared reconstruction error for each ECG separately.
            # Shape before mean: (batch, time, features)
            # Shape after mean:  (batch,)
            score = (x - reconstruction).pow(2).mean(dim=(1, 2))

            scores_for_masks.append(score)

        # Shape: (num_masks, batch)
        scores = torch.stack(scores_for_masks, dim=0)

        # Average over masks and return shape (batch, 1).
        return scores.mean(dim=0).unsqueeze(-1)



# ============================================================
# TranAD Anomaly Score Wrapper for SHAP
# ============================================================
class TranADAnomalyScoreWrapper(nn.Module):
    def forward(self, x):
        # Implementation for TranAD anomaly score wrapper
        pass