import random
import numpy as np
import torch


RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Seed the random generators used by this project."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
