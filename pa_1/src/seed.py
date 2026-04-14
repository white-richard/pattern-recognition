import os
import random

import numpy as np


def set_all_seeds(seed: int = 42) -> np.random.Generator:
    """Set deterministic seeds and return a seeded NumPy Generator."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)
