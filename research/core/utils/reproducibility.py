import os
import platform
import random
import sys
from typing import Dict

import numpy as np


def set_deterministic(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)


def runtime_manifest(seed: int) -> Dict:
    return {
        "seed": int(seed),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
        "tf_deterministic_ops": os.environ.get("TF_DETERMINISTIC_OPS", ""),
    }
