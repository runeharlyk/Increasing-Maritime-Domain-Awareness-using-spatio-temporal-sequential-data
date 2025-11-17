import random
import numpy as np
import torch

# Set seed for reproducibility
# References:
# https://docs.python.org/3/library/random.html
# https://numpy.org/doc/2.2/reference/random/generated/numpy.random.seed.html
# https://pytorch.org/docs/stable/notes/randomness.html

DEFAULT_SEED = 42


def default_rng(seed: int = DEFAULT_SEED):
    return np.random.default_rng(seed)


def set_seed(seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
