import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np


def timed_call(func: Callable, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def confidence_interval(values: Iterable[float], z: float = 1.96) -> Tuple[float, float, float]:
    arr = np.array(list(values), dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    margin = z * std / math.sqrt(max(len(arr), 1))
    return mean, std, margin


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def estimate_energy_joules(latency_s: float, assumed_watts: float = 15.0) -> float:
    return latency_s * assumed_watts


def memory_mb(array: np.ndarray) -> float:
    return array.nbytes / (1024 ** 2)
