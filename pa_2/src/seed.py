import os
import random

import numpy as np


def set_all_seeds(seed=42) -> None:
    """Set all seeds to make results reproducible."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
