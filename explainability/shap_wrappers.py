import torch
import torch.nn as nn


class AnoFormerAnomalyScoreWrapper(nn.Module):
    """
    Wraps AnoFormer as a scalar anomaly-score function for SHAP
    Output is the mean squared reconstruction error across all leads and timesteps
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
        
        self.generator = anoformer_model.Generator  # Only generator because the discriminator is used only during training not during anomaly detection

        if masking_rate is None:
            masking_rate = anoformer_model.Masking.RandomMaskingRate

        self.window_size = window_size
        self.masking_rate = masking_rate
        self.num_masks = num_masks

        self.generator.eval()

        for parameter in self.generator.parameters():
            parameter.requires_grad_(False)

        # Generate the masks once, making all later forward passes deterministic
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

        fixed_masks = fixed_masks.unsqueeze(-1)             
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
            masked_x = x * mask.unsqueeze(0)
            reconstruction = self.generator(masked_x)

            score = (x - reconstruction).pow(2).mean(dim=(1, 2))
            scores_for_masks.append(score)

        scores = torch.stack(scores_for_masks, dim=0)

        return scores.mean(dim=0).unsqueeze(-1)



class TranADAnomalyScoreWrapper(nn.Module):
    """
    Wraps TranAD as a scalar anomaly-score function for SHAP.
    Output is Phase-2 reconstruction MSE since only Phase two reconstruction is used for anomaly scoring
    """
    def __init__(self, tranad_model):
        super().__init__()

        self.Network = tranad_model.Network
        self.Network.eval()

        for parameter in self.Network.parameters():
            parameter.requires_grad_(False)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
             raise ValueError(
                "Expected input shape (batch, time, features)"
            )
        
        # two reconstructions, but we only need the second one
        _, reconstruction2 = self.Network(x)

        score = (reconstruction2 - x).pow(2).mean(dim=(1, 2))

        return score.unsqueeze(-1)  # shape (B, 1)
